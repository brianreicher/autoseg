from funlib.learn.torch.models import UNet, ConvPass
import numpy as np
import torch


class LSDModel(torch.nn.Module):
    def __init__(self, unet, num_fmaps):
        super(LSDModel, self).__init__()

        self.unet = unet
        self.lsd_head = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation="Sigmoid")

    def forward(self, input):
        x = self.unet(input)
        lsds = self.lsd_head(x)
        return lsds


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


in_channels = 1
num_fmaps = 12
fmap_inc_factor = 5

# add another downsampling factor for better net (but slower)
downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

unet = UNet(
    in_channels,
    num_fmaps,
    fmap_inc_factor,
    downsample_factors,
    constant_upsample=True,
)

model = LSDModel(unet, num_fmaps)
