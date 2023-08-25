from funlib.learn.torch.models import UNet, ConvPass
import numpy as np
import torch


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


in_channels = 1
num_fmaps = 12
fmap_inc_factor = 5

# add another downsampling factor for better net (but slower)
downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

n_diagonals = 8
neighborhood = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
    [4, 0, 0],
    [0, 4, 0],
    [0, 0, 4],
    [8, 0, 0],
    [0, 8, 0],
    [0, 0, 8],
]
pos_diag = np.round(
    n_diagonals * np.sin(np.linspace(0, np.pi, num=n_diagonals, endpoint=False))
)
neg_diag = np.round(
    n_diagonals * np.cos(np.linspace(0, np.pi, num=n_diagonals, endpoint=False))
)
stacked_diag = np.stack([0 * pos_diag, pos_diag, neg_diag], axis=-1)
neighborhood = np.concatenate([neighborhood, stacked_diag]).astype(np.int8)

unet = UNet(
    in_channels,
    num_fmaps,
    fmap_inc_factor,
    downsample_factors,
    constant_upsample=True,
)

model = MTLSDModel(unet, num_fmaps, num_affs=len(neighborhood))
