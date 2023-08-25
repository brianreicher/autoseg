import gunpowder as gp
import numpy as np
import torch
from funlib.learn.torch.models import UNet, ConvPass


voxel_size = gp.Coordinate((33,) * 3)

in_channels = 10
num_fmaps = 12
fmap_inc_factor = 5

downsample_factors = [(2, 2, 2), (2, 2, 2)]

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

model = torch.nn.Sequential(
    unet,
    ConvPass(num_fmaps, neighborhood.shape[0], [[1] * 3], activation="Sigmoid"),
)


increase = 8 * 10 * 2

input_shape = [132 + increase] * 3
output_shape = model.forward(torch.empty(size=[1, 10] + input_shape)).shape[-3:]

input_size = gp.Coordinate(input_shape) * voxel_size
output_size = gp.Coordinate(output_shape) * voxel_size

context = (input_size - output_size) // 2
