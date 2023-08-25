from model import *
import gunpowder as gp


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


voxel_size = gp.Coordinate((33,) * 3)

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

increase = 8 * 2

input_shape = [132 + increase] * 3
output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0][0].shape[1:]

input_size = gp.Coordinate(input_shape) * voxel_size
output_size = gp.Coordinate(output_shape) * voxel_size

context = ((input_size - output_size) / 2) * 4
