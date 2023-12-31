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


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):
        scaled = weights * (prediction - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss


def pipeline(iterations):
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_lsds_mask = gp.ArrayKey("GT_LSDS_MASK")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    unlabelled = gp.ArrayKey("UNLABELLED")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    voxel_size = gp.Coordinate((33,) * 3)

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5

    # add another downsampling factor for better net (but slower)
    downsample_factors = [(2, 2, 2), (2, 2, 2)]

    unet = UNet(
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        constant_upsample=True,
    )

    model = torch.nn.Sequential(
        unet, ConvPass(num_fmaps, 6, [[1] * 3], activation="Sigmoid")
    )

    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, betas=(0.95, 0.999))

    increase = 8

    input_shape = [132 + increase] * 3
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[1:]

    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    context = ((input_size - output_size) / 2) * 4

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_lsds, output_size)
    request.add(gt_lsds_mask, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(pred_lsds, output_size)

    source = gp.ZarrSource(
        "../../data/xpress-challenge.zarr",
        {
            raw: f"volumes/training_raw",
            labels: f"volumes/training_gt_rasters",
            labels_mask: f"volumes/training_labels_mask",
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            labels_mask: gp.ArraySpec(interpolatable=False),
        },
    )

    source += gp.Normalize(raw)
    source += gp.Pad(raw, None)
    source += gp.Pad(labels, context)
    source += gp.Pad(labels_mask, context)
    source += gp.RandomLocation(mask=labels_mask, min_masked=0.1)

    pipeline = source

    pipeline += gp.RandomProvider()

    pipeline += gp.ElasticAugment(
        control_point_spacing=[30, 30, 30],
        jitter_sigma=[2, 2, 2],
        rotation_interval=[0, math.pi / 2.0],
        subsample=8,
    )

    pipeline += gp.SimpleAugment()

    pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

    pipeline += DilatePoints(labels, dilations=6)

    pipeline += Relabel(labels)

    pipeline += Unlabel(labels, unlabelled)

    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        sigma=10 * 33,
        lsds_mask=gt_lsds_mask,
        unlabelled=unlabelled,
        downsample=2,
        components="345678",
    )

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)

    pipeline += gp.PreCache(cache_size=40, num_workers=10)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"input": raw},
        loss_inputs={0: pred_lsds, 1: gt_lsds, 2: gt_lsds_mask},
        outputs={0: pred_lsds},
        save_every=5000,
        log_dir="log",
    )

    pipeline += gp.Squeeze([raw, gt_lsds, pred_lsds])
    pipeline += gp.Squeeze([raw])

    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            labels: "labels",
            gt_lsds: "gt_lsds",
            unlabelled: "unlabelled",
            pred_lsds: "pred_lsds",
        },
        output_filename="batch_{iteration}.zarr",
        every=500,
    )

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":
    pipeline(50000)
