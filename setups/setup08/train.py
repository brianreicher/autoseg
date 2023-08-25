import gunpowder as gp
import logging
import math
import numpy as np
import random
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from lsd.train.gp import AddLocalShapeDescriptor
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    label,
    maximum_filter,
)
from skimage.measure import label as relabel
from tqdm import trange

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True


class DilatePoints(gp.BatchFilter):
    def __init__(self, labels, dilations=2):
        self.labels = labels
        self.dilations = dilations

    def process(self, batch, request):
        labels = batch[self.labels].data

        struct = generate_binary_structure(3, 1)

        dilated = binary_dilation(labels, structure=struct, iterations=self.dilations)

        batch[self.labels].data = dilated.astype(np.uint32)


class Relabel(gp.BatchFilter):
    def __init__(self, labels):
        self.labels = labels

    def process(self, batch, request):
        labels = batch[self.labels].data

        relabeled = relabel(labels, connectivity=1).astype(labels.dtype)

        batch[self.labels].data = relabeled


class Unlabel(gp.BatchFilter):
    def __init__(self, labels, unlabelled):
        self.labels = labels
        self.unlabelled = unlabelled

    def setup(self):
        self.provides(self.unlabelled, self.spec[self.labels].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.labels] = request[self.unlabelled].copy()

        return deps

    def process(self, batch, request):
        labels = batch[self.labels].data

        unlabelled = (labels > 0).astype(np.uint8)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.unlabelled].roi.copy()
        spec.dtype = np.uint8

        batch = gp.Batch()

        batch[self.unlabelled] = gp.Array(unlabelled, spec)

        return batch


class WeightedMTLSD_MSELoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super(WeightedMTLSD_MSELoss, self).__init__()

    def _calc_loss(self, prediction, target, weights):
        scaled = weights * (prediction - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
        self,
        pred_lsds=None,
        gt_lsds=None,
        lsds_weights=None,
        pred_affs=None,
        gt_affs=None,
        affs_weights=None,
    ):

        lsd_loss = self._calc_loss(pred_lsds, gt_lsds, lsds_weights)
        aff_loss = 0.7 * self._calc_loss(pred_affs, gt_affs, affs_weights)

        return lsd_loss + aff_loss


class MTLSDModel(torch.nn.Module):
    def __init__(self, unet, num_fmaps, num_affs=3):
        super(MTLSDModel, self).__init__()

        self.unet = unet
        self.aff_head = ConvPass(num_fmaps, num_affs, [[1, 1, 1]], activation="Sigmoid")
        self.lsd_head = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation="Sigmoid")

    def forward(self, input):
        x = self.unet(input)
        lsds = self.lsd_head(x)
        affs = self.aff_head(x)
        return lsds, affs


def pipeline(iterations, warmup=5000, save_every=1000):
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    gt_lsds_mask = gp.ArrayKey("GT_LSDS_MASK")
    gt_affs_mask = gp.ArrayKey("GT_AFFS_MASK")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    unlabelled = gp.ArrayKey("UNLABELLED")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    voxel_size = gp.Coordinate((33,) * 3)

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5

    # add another downsampling factor for better net (but slower)
    downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

    neighborhood = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        # [2,0,0],
        # [0,2,0],
        # [0,0,2],
        # [4,0,0],
        # [0,4,0],
        # [0,0,4],
        # [8,0,0],
        # [0,8,0],
        # [0,0,8]
    ]

    unet = UNet(
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        constant_upsample=True,
    )

    model = MTLSDModel(unet, num_fmaps)

    loss = WeightedMTLSD_MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, betas=(0.95, 0.999))

    increase = 8 * 2

    input_shape = [132 + increase] * 3
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0][0].shape[1:]

    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    context = ((input_size - output_size) / 2) * 4

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_lsds_mask, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(pred_lsds, output_size)
    request.add(pred_affs, output_size)

    rastered_source = gp.ZarrSource(
        "../../data/xpress-challenge.zarr",
        {
            raw: f"volumes/training_raw",
            labels: f"volumes/training_gt_rasters",
            labels_mask: f"volumes/training_raster_mask",
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            labels_mask: gp.ArraySpec(interpolatable=False),
        },
    )

    rastered_source += gp.RandomLocation(mask=labels_mask, min_masked=0.2)
    rastered_source += DilatePoints(labels, dilations=5)
    rastered_source += Relabel(labels)
    rastered_source += Unlabel(labels, unlabelled)
    rastered_source += gp.MergeProvider()

    voxel_source = gp.ZarrSource(
        "../../data/xpress-challenge.zarr",
        {
            raw: f"volumes/training_raw",
            labels: f"volumes/training_gt_labels",
            labels_mask: f"volumes/training_labels_mask",
            unlabelled: f"volumes/training_unlabelled_mask",
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            labels_mask: gp.ArraySpec(interpolatable=False),
            unlabelled: gp.ArraySpec(interpolatable=False),
        },
    )
    voxel_source += gp.Pad(unlabelled, context)
    voxel_source += gp.MergeProvider()
    voxel_source += gp.RandomLocation(mask=labels_mask, min_masked=0.2)

    mixed_source = (rastered_source, voxel_source) + gp.RandomProvider(
        probabilities=[0.8, 0.2]
    )

    def get_training_pipeline():
        training_pipeline = gp.Normalize(raw)
        training_pipeline += gp.Pad(raw, None)
        training_pipeline += gp.Pad(labels, context)
        training_pipeline += gp.Pad(labels_mask, context)
        # training_pipeline += gp.RandomLocation(mask=labels_mask, min_masked=0.1)
        # training_pipeline += gp.RandomLocation()
        # training_pipeline += gp.Reject(mask=labels_mask, min_masked=0.1)
        training_pipeline += gp.ElasticAugment(
            control_point_spacing=[30, 30, 30],
            jitter_sigma=[2, 2, 2],
            rotation_interval=[0, math.pi / 2.0],
            subsample=8,
        )
        training_pipeline += gp.SimpleAugment()
        training_pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)
        training_pipeline += AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            sigma=10 * 33,
            lsds_mask=gt_lsds_mask,
            unlabelled=unlabelled,
            downsample=2
            # components="123456789",
        )
        training_pipeline += gp.AddAffinities(
            neighborhood,
            labels,
            gt_affs,
            labels_mask,
            unlabelled,
            gt_affs_mask,
        )

        training_pipeline += gp.Unsqueeze([raw])
        training_pipeline += gp.Stack(3)

        training_pipeline += gp.PreCache(cache_size=40, num_workers=10)

        training_pipeline += gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs={"input": raw},
            loss_inputs={
                0: pred_lsds,
                1: gt_lsds,
                2: gt_lsds_mask,
                3: pred_affs,
                4: gt_affs,
                5: gt_affs_mask,
            },
            outputs={0: pred_lsds, 1: pred_affs},
            save_every=save_every,
            log_dir="log",
        )

        # training_pipeline += gp.Squeeze([raw, gt_lsds, pred_lsds, gt_affs, pred_affs])
        training_pipeline += gp.Squeeze([raw], 1)

        training_pipeline += gp.Snapshot(
            dataset_names={
                raw: "raw",
                labels: "labels",
                gt_lsds: "gt_lsds",
                unlabelled: "unlabelled",
                pred_lsds: "pred_lsds",
                gt_affs: "gt_affs",
                pred_affs: "pred_affs",
            },
            output_filename="batch_{iteration}.zarr",
            every=save_every,
        )

        return training_pipeline

    # First iterations are warmup on voxel data
    if warmup > 0:
        training_pipeline = get_training_pipeline()
        pipeline = rastered_source + training_pipeline
        # pipeline = voxel_source + training_pipeline
        with gp.build(pipeline):
            for i in trange(warmup):
                pipeline.request_batch(request)

    # Then switch to mixed data
    training_pipeline = get_training_pipeline()
    pipeline = mixed_source + training_pipeline
    # pipeline = rastered_source + training_pipeline
    with gp.build(pipeline):
        for i in trange(iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":
    # pipeline(1, 1, 5000)
    pipeline(50000, 50000, 5000)
