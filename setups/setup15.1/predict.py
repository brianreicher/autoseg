import gunpowder as gp
import logging
import math
import numpy as np
import os
import sys
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass

logging.basicConfig(level=logging.INFO)

from model import *


def predict(lsds_file, lsds_dataset, out_file, out_dataset):
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    increase = 8 * 10

    input_shape = [132 + increase] * 3
    output_shape = model.forward(torch.empty(size=[1, 10] + input_shape))[0].shape[1:]

    print(output_shape)

    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    context = (input_size - output_size) // 2

    model.eval()

    scan_request = gp.BatchRequest()

    scan_request.add(pred_lsds, input_size)
    scan_request.add(pred_affs, output_size)

    source = gp.ZarrSource(
        lsds_file,
        {pred_lsds: lsds_dataset},
        {pred_lsds: gp.ArraySpec(interpolatable=True)},
    )

    with gp.build(source):
        total_input_roi = source.spec[pred_lsds].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    f = zarr.open(out_file, "w")

    ds = f.create_dataset(
        out_dataset,
        shape=[3] + list(total_output_roi.get_shape() / voxel_size),
        dtype=np.uint8,
    )

    ds.attrs["resolution"] = voxel_size
    ds.attrs["offset"] = total_output_roi.get_offset()

    predict = gp.torch.Predict(
        model,
        checkpoint=f"model_checkpoint_2000",
        inputs={"input": pred_lsds},
        outputs={
            0: pred_affs,
        },
    )

    write = gp.ZarrWrite(
        dataset_names={
            pred_affs: out_dataset,
        },
        output_filename=out_file,
    )

    scan = gp.Scan(scan_request)

    pipeline = (
        source
        + gp.Normalize(pred_lsds)
        + gp.Unsqueeze([pred_lsds])
        + predict
        + gp.Squeeze([pred_affs])
        + gp.IntensityScaleShift(pred_affs, 255, 0)
        + write
        + scan
    )

    predict_request = gp.BatchRequest()

    predict_request[pred_lsds] = total_input_roi
    predict_request[pred_affs] = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)


if __name__ == "__main__":
    lsds_file = "../setup13/test_prediction.zarr"
    lsds_dataset = "pred_lsds"
    out_file = "test_prediction.zarr"
    out_dataset = "pred_affs"

    predict(lsds_file, lsds_dataset, out_file, out_dataset)
