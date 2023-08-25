from model import *

import numpy as np
import gunpowder as gp
from skimage.measure import label as relabel
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    # distance_transform_edt,
    generate_binary_structure,
    # label,
    # maximum_filter,
)


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


class WeightedLSD_MSELoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super(WeightedLSD_MSELoss, self).__init__()

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
    ):

        lsd_loss = self._calc_loss(pred_lsds, gt_lsds, lsds_weights)

        return lsd_loss


class ChangeBackground(gp.BatchFilter):
    def __init__(self, labels):
        self.labels = labels

    def process(self, batch, request):
        labels = batch[self.labels].data

        labels[labels == 0] = np.max(labels) + 1

        batch[self.labels].data = labels


class DilatePoints(gp.BatchFilter):
    def __init__(self, labels, dilations=2):
        self.labels = labels
        self.dilations = dilations

    def process(self, batch, request):
        labels = batch[self.labels].data

        struct = generate_binary_structure(3, 1)

        dilated = binary_dilation(labels, structure=struct, iterations=self.dilations)

        batch[self.labels].data = dilated.astype(np.uint32)


class ErodePoints(gp.BatchFilter):
    def __init__(self, labels, erosions=2):
        self.labels = labels
        self.erosions = erosions

    def process(self, batch, request):
        labels = batch[self.labels].data

        struct = generate_binary_structure(3, 1)

        eroded = binary_erosion(labels, structure=struct, iterations=self.erosions)

        batch[self.labels].data = eroded.astype(np.uint32)


class Relabel(gp.BatchFilter):
    def __init__(self, labels):
        self.labels = labels

    def process(self, batch, request):
        labels = batch[self.labels].data

        relabeled = relabel(labels, connectivity=1).astype(labels.dtype)

        batch[self.labels].data = relabeled


voxel_size = gp.Coordinate((33,) * 3)

increase = 8 * 3

input_shape = [132 + increase] * 3
output_shape = model.forward(torch.empty(size=[1, 1] + input_shape)).shape[-3:]

input_size = gp.Coordinate(input_shape) * voxel_size
output_size = gp.Coordinate(output_shape) * voxel_size

context = ((input_size - output_size) / 2) * 4

from aff_model import (
    input_size as aff_input_size,
    output_size as aff_output_size,
    context as aff_context,
    model as aff_model,
)

raw_file = "../../data/xpress-challenge.zarr"
raw_dataset = "volumes/training_raw"
out_file = "./predictions.zarr"
