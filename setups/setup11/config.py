from model import *
import numpy as np
import gunpowder as gp


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
    def __init__(self, aff_lambda=0.7) -> None:
        super(WeightedMTLSD_MSELoss, self).__init__()
        self.aff_lambda = aff_lambda

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
        aff_loss = self.aff_lambda * self._calc_loss(pred_affs, gt_affs, affs_weights)

        return lsd_loss + aff_loss


class ChangeBackground(gp.BatchFilter):
    def __init__(self, labels):
        self.labels = labels

    def process(self, batch, request):
        labels = batch[self.labels].data

        labels[labels == 0] = np.max(labels) + 1

        batch[self.labels].data = labels


voxel_size = gp.Coordinate((33,) * 3)

increase = 8 * 2

input_shape = [132 + increase] * 3
output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0][0].shape[1:]

input_size = gp.Coordinate(input_shape) * voxel_size
output_size = gp.Coordinate(output_shape) * voxel_size

context = ((input_size - output_size) / 2) * 4
