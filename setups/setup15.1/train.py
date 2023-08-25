import gunpowder as gp
import logging
import math
import numpy as np
import random
import torch
import zarr
from lsd.train.gp import AddLocalShapeDescriptor
from lsd.train import LsdExtractor
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
)
from skimage.measure import label
from skimage.morphology import disk

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

from model import *
"""
Generate a random binary structure and perform a random number of dilations on each 2D slice of the labels array.
Relabel the labels array to ensure that each point has a unique label.
Update the labels data of the Gunpowder batch.
Overall, this code is simulating the presence of objects in an image by randomly placing points and then applying a series of transformations to those points to generate larger, more complex objects.

"""

class CreatePoints(gp.BatchFilter):
    """
    Creates a random array of points to dialates them to similate the presence of objects in an image slice
    """
    def __init__(
        self,
        labels,
    ):
        self.labels = labels

    def process(self, batch, request):
        """
        Process a Gunpowder batch and generate points/preform dialations.

        Args:
            batch : Gunpowder batch to process.
        """

        # extract points, shape from batch
        labels = batch[self.labels].data
        shape = labels.shape

        spec = batch[self.labels].spec

        # Generates a random number of points to simulate randomly placed objects 
        # different numbers simulate more or less objects
        num_points = random.randint(25, 100)
        # num_points = 30

        for n in range(num_points):
            z = random.randint(1, labels.shape[0] - 1)
            y = random.randint(1, labels.shape[1] - 1)
            x = random.randint(1, labels.shape[2] - 1)

            labels[z, y, x] = 1

        structs = [generate_binary_structure(2, 2), disk(random.randint(1, 5))]

        for z in range(labels.shape[0]):
            # different numbers will simulate larger or smaller objects
            struct = random.choice(structs)
            dilations = random.randint(1, 10)

            dilated = binary_dilation(labels[z], structure=struct, iterations=dilations)

            labels[z] = dilated.astype(labels.dtype)

        # relabel
        labels = label(labels, connectivity=2).astype(labels.dtype)

        batch[self.labels].data = labels


class Relabel(gp.BatchFilter):
    def __init__(self, labels):
        self.labels = labels

    def process(self, batch, request):
        labels = batch[self.labels].data

        relabeled = label(labels, connectivity=1).astype(labels.dtype)

        batch[self.labels].data = relabeled


class ExpandLabels(gp.BatchFilter):
    def __init__(self, labels, background=0):
        self.labels = labels
        self.background = background

    def process(self, batch, request):
        labels_data = batch[self.labels].data
        distance = labels_data.shape[0]

        distances, indices = distance_transform_edt(
            labels_data == self.background, return_indices=True
        )

        expanded_labels = np.zeros_like(labels_data)

        dilate_mask = distances <= distance

        masked_indices = [
            dimension_indices[dilate_mask] for dimension_indices in indices
        ]

        nearest_labels = labels_data[tuple(masked_indices)]

        expanded_labels[dilate_mask] = nearest_labels

        batch[self.labels].data = expanded_labels


class ChangeBackground(gp.BatchFilter):
    def __init__(self, labels):
        self.labels = labels

    def process(self, batch, request):
        labels = batch[self.labels].data

        labels[labels == 0] = np.max(labels) + 1

        batch[self.labels].data = labels


class SmoothLSDs(gp.BatchFilter):
    def __init__(self, lsds):
        self.lsds = lsds

    def process(self, batch, request):
        lsds = batch[self.lsds].data

        sigma = random.uniform(0.5, 2.0)

        for z in range(lsds.shape[1]):
            lsds_sec = lsds[:, z]

            lsds[:, z] = np.array(
                [
                    gaussian_filter(lsds_sec[i], sigma=sigma)
                    for i in range(lsds_sec.shape[0])
                ]
            ).astype(lsds_sec.dtype)

        batch[self.lsds].data = lsds


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


class ZerosSource(gp.BatchProvider):
    def __init__(self, datasets, shape=None, dtype=np.uint64, array_specs=None):
        self.datasets = datasets

        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.shape = shape if shape is not None else gp.Coordinate((200, 200, 200))
        self.dtype = dtype

        # number of spatial dimensions
        self.ndims = None

    def setup(self):
        for array_key, ds_name in self.datasets.items():
            if array_key in self.array_specs:
                spec = self.array_specs[array_key].copy()
            else:
                spec = gp.ArraySpec()

            if spec.voxel_size is None:
                voxel_size = gp.Coordinate((1,) * len(self.shape))
                spec.voxel_size = voxel_size

            self.ndims = len(spec.voxel_size)

            if spec.roi is None:
                offset = gp.Coordinate((0,) * self.ndims)
                spec.roi = gp.Roi(offset, self.shape * spec.voxel_size)

            if spec.dtype is not None:
                assert spec.dtype == self.dtype
            else:
                spec.dtype = self.dtype

            if spec.interpolatable is None:
                spec.interpolatable = spec.dtype in [
                    np.float,
                    np.float32,
                    np.float64,
                    np.float128,
                    np.uint8,  # assuming this is not used for labels
                ]

            self.provides(array_key, spec)

    def provide(self, request):
        batch = gp.Batch()

        for array_key, request_spec in request.array_specs.items():
            voxel_size = self.spec[array_key].voxel_size

            # scale request roi to voxel units
            dataset_roi = request_spec.roi / voxel_size

            # shift request roi into dataset
            dataset_roi = (
                dataset_roi - self.spec[array_key].roi.get_offset() / voxel_size
            )

            # create array spec
            array_spec = self.spec[array_key].copy()
            array_spec.roi = request_spec.roi

            # add array to batch
            batch.arrays[array_key] = gp.Array(
                np.zeros(self.shape, self.dtype), array_spec
            )

        return batch


def pipeline(iterations):
    zeros = gp.ArrayKey("ZEROS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, betas=(0.95, 0.999))

    request = gp.BatchRequest()

    request.add(zeros, input_size)
    request.add(gt_lsds, input_size)
    request.add(gt_affs, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)

    source = ZerosSource(
        {zeros: "zeros"},
        shape=gp.Coordinate(input_shape),
        array_specs={
            zeros: gp.ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size,
            )
        },
    )

    source += gp.Pad(zeros, context)

    pipeline = source

    pipeline += CreatePoints(zeros)

    pipeline += ExpandLabels(zeros)

    pipeline += gp.SimpleAugment()

    pipeline += AddLocalShapeDescriptor(
        zeros,
        gt_lsds,
        sigma=4 * 33,
        downsample=2,
    )

    pipeline += gp.NoiseAugment(gt_lsds)

    pipeline += gp.IntensityAugment(gt_lsds, 0.9, 1.1, -0.1, 0.1)

    pipeline += SmoothLSDs(gt_lsds)

    pipeline += ChangeBackground(zeros)

    pipeline += Relabel(zeros)

    pipeline += gp.GrowBoundary(zeros, steps=1)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=zeros,
        affinities=gt_affs,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights)

    pipeline += gp.Stack(1)

    pipeline += gp.PreCache(cache_size=40, num_workers=20)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"input": gt_lsds},
        loss_inputs={0: pred_affs, 1: gt_affs, 2: affs_weights},
        outputs={0: pred_affs},
        save_every=2000,
    )

    pipeline += gp.Squeeze([gt_lsds, gt_affs, pred_affs])

    pipeline += gp.Snapshot(
        dataset_names={
            zeros: "zeros",
            gt_lsds: "gt_lsds",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
        },
        output_filename="batch_{iteration}.zarr",
        every=2000,
    )

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":
    pipeline(10000)
