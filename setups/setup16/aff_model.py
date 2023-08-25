import gunpowder as gp
import numpy as np
import torch
from funlib.learn.torch.models import UNet, ConvPass


voxel_size = gp.Coordinate((33,) * 3)

in_channels = 10
num_fmaps = 12
fmap_inc_factor = 5

downsample_factors = [(2, 2, 2), (2, 2, 2)]

neighborhood = [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [-2, 0, 0],
        [0, -2, 0],
        [0, 0, -2],
        [-4, 0, 0],
        [0, -4, 0],
        [0, 0, -4],
        [-8, 0, 0],
        [0, -8, 0],
        [0, 0, -8],
        [0, -3, -7],
        [0, -6, -6],
        [0, -7, -3],
        [0, -7, 3],
        [0, -6, 6],
        [0, -3, 7]
]
neighborhood = np.array(neighborhood)

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
#output_shape = model.forward(torch.empty(size=[1, 10] + input_shape)).shape[-3:]
output_shape = [252] * 3

input_size = gp.Coordinate(input_shape) * voxel_size
output_size = gp.Coordinate(output_shape) * voxel_size

context = (input_size - output_size) // 2
