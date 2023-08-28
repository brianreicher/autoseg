import numpy as np
import gunpowder as gp
import torch
from funlib.learn.torch.models import UNet, ConvPass
import random
from scipy.ndimage import gaussian_filter


class WeightedMTLSD_MSELoss(torch.nn.MSELoss):
    def __init__(self, aff_lambda=1.0) -> None:
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


class MTLSDModel(torch.nn.Module):
    def __init__(self, unet, num_fmaps):
        super(MTLSDModel, self).__init__()

        self.unet = unet
        self.lsd_head = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation="Sigmoid")
        self.aff_head = ConvPass(num_fmaps, 3, [[1, 1, 1]], activation="Sigmoid")

    def forward(self, input):
        x = self.unet(input)
        lsds = self.lsd_head(x[0])
        affs = self.aff_head(x[1])
        return lsds, affs


class SmoothArray(gp.BatchFilter):
    def __init__(self, array, blur_range):
        self.array = array
        self.range = blur_range

    def process(self, batch, request):

        array = batch[self.array].data

        assert len(array.shape) == 3

        # different numbers will simulate noisier or cleaner array
        sigma = random.uniform(self.range[0], self.range[1])

        for z in range(array.shape[0]):
            array_sec = array[z]

            array[z] = np.array(
                    gaussian_filter(array_sec, sigma=sigma)
            ).astype(array_sec.dtype)

        batch[self.array].data = array


class RandomNoiseAugment(gp.BatchFilter):
    def __init__(self, array, seed=None, clip=True, **kwargs):
        self.array = array
        self.seed = seed
        self.clip = clip
        self.kwargs = kwargs

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]

        mode = random.choice(["gaussian","poisson","none", "none"])

        if mode != "none":
            assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, "Noise augmentation requires float types for the raw array (not " + str(raw.data.dtype) + "). Consider using Normalize before."
            if self.clip:
                assert raw.data.min() >= -1 and raw.data.max() <= 1, "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."


in_channels = 1
num_fmaps = 12
fmap_inc_factor = 3
downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

unet = UNet(
    in_channels,
    num_fmaps,
    fmap_inc_factor,
    downsample_factors,
    constant_upsample=True,
    num_heads=2
)
