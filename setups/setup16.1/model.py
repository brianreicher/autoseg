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


in_channels = 1
num_fmaps = 12
fmap_inc_factor = 5

# add another downsampling factor for better net (but slower)
downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

from aff_model import n_diagonals, neighborhood

unet = UNet(
    in_channels,
    num_fmaps,
    fmap_inc_factor,
    downsample_factors,
    constant_upsample=True,
)

model = LSDModel(unet, num_fmaps)
