import gunpowder as gp
import logging
import math
import numpy as np
import random
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from lsd.train.gp import AddLocalShapeDescriptor
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    label,
    maximum_filter,
)
from skimage.measure import label as relabel
from tqdm import trange

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

from config import *


def pipeline(iterations, warmup=5000, save_every=1000):
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    gt_lsds_mask = gp.ArrayKey("GT_LSDS_MASK")
    gt_affs_mask = gp.ArrayKey("GT_AFFS_MASK")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    unlabelled = gp.ArrayKey("UNLABELLED")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_lsds_mask, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(pred_lsds, output_size)
    request.add(pred_affs, output_size)

    voxel_source = gp.ZarrSource(
        "../../data/xpress-challenge.zarr",
        {
            raw: f"volumes/training_raw",
            labels: f"volumes/training_gt_labels",
            labels_mask: f"volumes/training_labels_mask",
            unlabelled: f"volumes/training_unlabelled_mask",
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            labels_mask: gp.ArraySpec(interpolatable=False),
            unlabelled: gp.ArraySpec(interpolatable=False),
        },
    )
    voxel_source += gp.Pad(unlabelled, context)
    voxel_source += gp.MergeProvider()
    voxel_source += gp.RandomLocation(mask=labels_mask, min_masked=0.2)

    def get_training_pipeline(aff_lambda=0.7, init_lr=0.5e-4, betas=(0.95, 0.999)):
        loss = WeightedMTLSD_MSELoss(aff_lambda=aff_lambda)
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=betas)

        training_pipeline = gp.Normalize(raw)
        training_pipeline += gp.Pad(raw, None)
        training_pipeline += gp.Pad(labels, context)
        training_pipeline += gp.Pad(labels_mask, context)
        # training_pipeline += gp.RandomLocation(mask=labels_mask, min_masked=0.1)
        # training_pipeline += gp.RandomLocation()
        # training_pipeline += gp.Reject(mask=labels_mask, min_masked=0.1)
        training_pipeline += gp.ElasticAugment(
            control_point_spacing=[30, 30, 30],
            jitter_sigma=[2, 2, 2],
            rotation_interval=[0, math.pi / 2.0],
            subsample=8,
        )
        training_pipeline += gp.SimpleAugment()
        training_pipeline += gp.NoiseAugment(raw)
        training_pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)
        training_pipeline += AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            sigma=10 * 33,
            lsds_mask=gt_lsds_mask,
            unlabelled=unlabelled,
            downsample=2
            # components="123456789",
        )
        training_pipeline += gp.AddAffinities(
            neighborhood,
            labels,
            gt_affs,
            labels_mask,
            unlabelled,
            gt_affs_mask,
        )

        training_pipeline += gp.Unsqueeze([raw])
        training_pipeline += gp.Stack(3)

        training_pipeline += gp.PreCache(cache_size=40, num_workers=10)

        training_pipeline += gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs={"input": raw},
            loss_inputs={
                0: pred_lsds,
                1: gt_lsds,
                2: gt_lsds_mask,
                3: pred_affs,
                4: gt_affs,
                5: gt_affs_mask,
            },
            outputs={0: pred_lsds, 1: pred_affs},
            save_every=save_every,
            log_dir="log",
        )

        # training_pipeline += gp.Squeeze([raw, gt_lsds, pred_lsds, gt_affs, pred_affs])
        training_pipeline += gp.Squeeze([raw], 1)

        training_pipeline += gp.Snapshot(
            dataset_names={
                raw: "raw",
                labels: "labels",
                gt_lsds: "gt_lsds",
                unlabelled: "unlabelled",
                pred_lsds: "pred_lsds",
                gt_affs: "gt_affs",
                pred_affs: "pred_affs",
            },
            output_filename="batch_{iteration}.zarr",
            every=save_every,
        )

        return training_pipeline

    # Start with affs having 70% weight
    training_pipeline = get_training_pipeline(init_lr=0.5e-3)
    pipeline = voxel_source + training_pipeline
    with gp.build(pipeline):
        for i in trange(warmup):
            pipeline.request_batch(request)

    # Then make affs have equal weight
    training_pipeline = get_training_pipeline(aff_lambda=1.0, init_lr=0.5e-4)
    pipeline = voxel_source + training_pipeline
    with gp.build(pipeline):
        for i in trange(iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":
    pipeline(10000, 10000, 5000)
