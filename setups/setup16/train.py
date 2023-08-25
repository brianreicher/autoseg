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
from predict import predict_task
from correct_seg import correct_segmentation
from segment_correct_blocks import segment_correct_blocks

raw_file = "../../data/xpress-challenge.zarr"
raw_dataset = "volumes/training_raw"
out_file = "./predictions.zarr"


def get_segmentation():
    predict_task(
        raw_file=raw_file,
        raw_dataset=raw_dataset,
        out_file=out_file,
        out_datasets=["pred_lsds"],
        out_shapes=[10],
    )
    predict_task(
        raw_file=out_file,
        raw_dataset="pred_lsds",
        raw_shape=10,
        out_file=out_file,
        out_datasets=["pred_affs"],
        out_shapes=[len(neighborhood)],
        model=aff_model,
        # model_path="../setup15.1/latest",
        model_path="../setup15.1/model_checkpoint_4000",
        input_size=aff_input_size,
        output_size=aff_output_size,
        context=aff_context,
    )
    # segment()
    # correct_segmentation()
    return segment_correct_blocks()


def pipeline(iterations, warmup=5000, save_every=1000):
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_lsds_mask = gp.ArrayKey("GT_LSDS_MASK")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    unlabelled = gp.ArrayKey("UNLABELLED")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

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
                # labels: f"pred_seg",
                labels: f"frag_seg",
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

    rastered_source = gp.ZarrSource(
        "../../data/xpress-challenge.zarr",
        {
            raw: f"volumes/training_raw",
            labels: f"volumes/training_gt_rasters",
            labels_mask: f"volumes/training_raster_mask",
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            labels_mask: gp.ArraySpec(interpolatable=False),
        },
    )

    rastered_source += gp.RandomLocation(mask=labels_mask, min_masked=0.2)
    rastered_source += DilatePoints(labels, dilations=5)
    rastered_source += Relabel(labels)
    rastered_source += Unlabel(labels, unlabelled)
    rastered_source += gp.MergeProvider()

    def get_training_pipeline():
        request = gp.BatchRequest()

        request.add(raw, input_size)
        request.add(labels, output_size)
        request.add(gt_lsds, output_size)
        request.add(gt_lsds_mask, output_size)
        request.add(labels_mask, output_size)
        request.add(unlabelled, output_size)
        request.add(pred_lsds, output_size)

        loss = WeightedLSD_MSELoss()

        training_pipeline = gp.Normalize(raw)
        training_pipeline += gp.Pad(raw, None)
        training_pipeline += gp.Pad(labels, context)
        training_pipeline += gp.Pad(labels_mask, context)
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
            sigma=4 * 33,
            lsds_mask=gt_lsds_mask,
            unlabelled=unlabelled,
            downsample=2,
        )

        training_pipeline += gp.IntensityScaleShift(raw, 2, -1)
        training_pipeline += gp.Unsqueeze([raw])
        training_pipeline += gp.Stack(1)

        training_pipeline += gp.PreCache(cache_size=40, num_workers=20)

        training_pipeline += gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs={"input": raw},
            loss_inputs={0: pred_lsds, 1: gt_lsds, 2: gt_lsds_mask},
            outputs={0: pred_lsds},
            save_every=save_every,
            log_dir="log",
            spawn_subprocess=True,
        )

        # training_pipeline += gp.Squeeze([raw, gt_lsds, pred_lsds, gt_affs, pred_affs])
        training_pipeline += gp.Squeeze([raw], 1)
        training_pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

        training_pipeline += gp.Snapshot(
            dataset_names={
                raw: "raw",
                labels: "labels",
                gt_lsds: "gt_lsds",
                pred_lsds: "pred_lsds",
            },
            output_filename="batch_{iteration}.zarr",
            every=save_every,
        )

        return training_pipeline, request

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, betas=(0.95, 0.999))

    # First iterations are warmup on voxel data
    if (
        warmup is None
    ):  # Allows to do initial segmentation with existing model checkpoints
        # Make segmentation predictions
        get_segmentation()
        model.train()
    elif warmup > 0:
        training_pipeline, request = get_training_pipeline()
        pipeline = (
            # (gt_source, rastered_source)
            # + gp.RandomProvider([0.9, 0.1])
            gt_source
            + training_pipeline
        )
        with gp.build(pipeline):
            for i in trange(warmup):
                pipeline.request_batch(request)

        # Make segmentation predictions
        get_segmentation()
        model.train()

    # Add segmentation predictions to training pipeline
    # Then repeat, scaling up the prediction usage
    for ratio in [0.01, 0.01, 0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.4, 0.4, 0.8]:
        training_pipeline, request = get_training_pipeline()
        pipeline = (gt_source, predicted_source) + gp.RandomProvider(
            probabilities=[1 - ratio, ratio]
        )
        pipeline += training_pipeline
        with gp.build(pipeline):
            for i in trange(iterations):
                pipeline.request_batch(request)

        # Make segmentation predictions
        get_segmentation()
        model.train()


if __name__ == "__main__":
    # pipeline(10000, 20000, 10000)
    # pipeline(10000, 0, 10000)
    pipeline(10000, None, 1000)
