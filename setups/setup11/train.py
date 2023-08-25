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
from segment import segment
from predict import predict


raw_file = "../../data/xpress-challenge.zarr"
raw_dataset = "volumes/training_raw"
out_file = "./predictions.zarr"


def pipeline(iterations, warmup=5000, save_every=1000):
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    gt_lsds_mask = gp.ArrayKey("GT_LSDS_MASK")
    gt_affs_mask = gp.ArrayKey("GT_AFFS_MASK")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    # lsds_weights = gp.ArrayKey("LSDS_WEIGHTS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    unlabelled = gp.ArrayKey("UNLABELLED")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    predicted_source = (
        gp.ZarrSource(
            raw_file,
            {
                raw: raw_dataset,
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
            },
        ),
        gp.ZarrSource(
            out_file,
            {
                labels: f"pred_seg",
                labels_mask: f"pred_labels_mask",
                unlabelled: f"pred_unlabelled_mask",
            },
            {
                labels: gp.ArraySpec(interpolatable=False),
                labels_mask: gp.ArraySpec(interpolatable=False),
                unlabelled: gp.ArraySpec(interpolatable=False),
            },
        ),
    ) + gp.MergeProvider()
    predicted_source += gp.Pad(unlabelled, context)
    predicted_source += gp.MergeProvider()
    # predicted_source += Unlabel(labels, unlabelled)
    predicted_source += gp.RandomLocation(mask=labels_mask, min_masked=0.2)

    gt_source = gp.ZarrSource(
        raw_file,
        {
            raw: raw_dataset,
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
    gt_source += gp.Pad(unlabelled, context)
    gt_source += gp.MergeProvider()
    gt_source += gp.RandomLocation(mask=labels_mask, min_masked=0.2)

    def get_training_pipeline(aff_lambda=0.7, change_background=True):
        request = gp.BatchRequest()

        request.add(raw, input_size)
        request.add(labels, output_size)
        request.add(gt_lsds, output_size)
        request.add(gt_affs, output_size)
        request.add(gt_lsds_mask, output_size)
        request.add(labels_mask, output_size)
        request.add(unlabelled, output_size)
        request.add(pred_lsds, output_size)
        request.add(pred_affs, output_size)

        loss = WeightedMTLSD_MSELoss(aff_lambda=aff_lambda)

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

        if change_background:
            request.add(affs_weights, output_size)
            # request.add(lsds_weights, output_size)

            training_pipeline += ChangeBackground(labels)
            training_pipeline += gp.GrowBoundary(labels)

            training_pipeline += AddLocalShapeDescriptor(
                labels,
                gt_lsds,
                sigma=10 * 33,
                downsample=2,
                lsds_mask=gt_lsds_mask,
            )
            training_pipeline += gp.AddAffinities(
                neighborhood,
                labels,
                gt_affs,
            )
            # training_pipeline += gp.BalanceLabels(
            #     gt_lsds,
            #     lsds_weights,
            #     # num_classes=255, #?
            # )
            training_pipeline += gp.BalanceLabels(
                gt_affs,
                affs_weights,
            )

        else:
            request.add(gt_affs_mask, output_size)
            # request.add(gt_lsds_mask, output_size)

            training_pipeline += AddLocalShapeDescriptor(
                labels,
                gt_lsds,
                sigma=10 * 33,
                lsds_mask=gt_lsds_mask,
                unlabelled=unlabelled,
                downsample=2,
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

        if change_background:
            training_pipeline += gp.torch.Train(
                model,
                loss,
                optimizer,
                inputs={"input": raw},
                loss_inputs={
                    0: pred_lsds,
                    1: gt_lsds,
                    # 2: lsds_weights,
                    2: gt_lsds_mask,
                    3: pred_affs,
                    4: gt_affs,
                    5: affs_weights,
                },
                outputs={0: pred_lsds, 1: pred_affs},
                save_every=save_every,
                log_dir="log",
                spawn_subprocess=True,
            )
        else:
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
                spawn_subprocess=True,
            )

        # training_pipeline += gp.Squeeze([raw, gt_lsds, pred_lsds, gt_affs, pred_affs])
        training_pipeline += gp.Squeeze([raw], 1)

        training_pipeline += gp.Snapshot(
            dataset_names={
                raw: "raw",
                labels: "labels",
                unlabelled: "unlabelled",
                gt_lsds: "gt_lsds",
                pred_lsds: "pred_lsds",
                gt_affs: "gt_affs",
                pred_affs: "pred_affs",
            },
            output_filename="batch_{iteration}.zarr",
            every=save_every,
        )

        return training_pipeline, request

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-3, betas=(0.95, 0.999))

    # Start with affs having 70% weight
    # First iterations are warmup on voxel data
    if warmup is None:
        # Make segmentation predictions
        predict(raw_file, raw_dataset, out_file)
        model.train()
        segment()
    elif warmup > 0:
        training_pipeline, request = get_training_pipeline()
        pipeline = gt_source + training_pipeline
        with gp.build(pipeline):
            for i in trange(warmup):
                pipeline.request_batch(request)

        # Make segmentation predictions
        predict(raw_file, raw_dataset, out_file)
        model.train()
        segment()

    # Add segmentation predictions to training pipeline
    # Then repeat, scaling up the prediction usage
    for ratio in [0.05, 0.25, 0.5, 0.75, 0.95]:
        training_pipeline, request = get_training_pipeline()
        pipeline = (gt_source, predicted_source) + gp.RandomProvider(
            probabilities=[1 - ratio, ratio]
        )
        pipeline += training_pipeline
        with gp.build(pipeline):
            for i in trange(iterations):
                pipeline.request_batch(request)

        # Make segmentation predictions
        predict(raw_file, raw_dataset, out_file)
        model.train()
        segment()

    # Then repeat, without relabeling the background
    for ratio in [0.05, 0.25, 0.5, 0.75, 0.95]:
        training_pipeline, request = get_training_pipeline(change_background=False)
        pipeline = (gt_source, predicted_source) + gp.RandomProvider(
            probabilities=[1 - ratio, ratio]
        )
        pipeline += training_pipeline
        with gp.build(pipeline):
            for i in trange(iterations):
                pipeline.request_batch(request)

        # Make segmentation predictions
        predict(raw_file, raw_dataset, out_file)
        model.train()
        segment()

    # Then make affs have equal weight
    training_pipeline, request = get_training_pipeline(
        aff_lambda=1.0, change_background=False
    )
    pipeline = (gt_source, predicted_source) + gp.RandomProvider()
    pipeline += training_pipeline
    with gp.build(pipeline):
        for i in trange(iterations):
            pipeline.request_batch(request)

    # Make segmentation predictions
    predict(raw_file, raw_dataset, out_file)
    # model.train()
    segment()


if __name__ == "__main__":
    # pipeline(5, 5, 5)
    # pipeline(5, 0, 5)
    # pipeline(10000, 20000, 5000)
    pipeline(10000, None, 5000)
