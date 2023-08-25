from glob import glob
import logging
import math
import numpy as np
import os
import sys
import zarr
from config import *

logging.basicConfig(level=logging.INFO)

increase *= 3

input_shape = [132 + increase] * 3
output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0][0].shape[1:]

input_size = gp.Coordinate(input_shape) * voxel_size
output_size = gp.Coordinate(output_shape) * voxel_size

context = ((input_size - output_size) / 2) * 4


def predict(
    raw_file,
    raw_dataset,
    out_file,
    out_datasets=["pred_lsds", "pred_affs"],
    out_shapes=[10, len(neighborhood)],
    model_path="latest",
):
    if model_path == "latest":
        model_path = glob("./model_checkpoint_*")
        model_path.sort(key=os.path.getmtime)
        model_path = os.path.abspath(model_path[-1])

    raw = gp.ArrayKey("RAW")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    model.eval()

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_lsds, output_size)
    scan_request.add(pred_affs, output_size)

    source = gp.ZarrSource(
        raw_file, {raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    f = zarr.open(out_file, "w")
    for out_dataset, output_shape in zip(out_datasets, out_shapes):
        ds = f.create_dataset(
            out_dataset,
            shape=[output_shape]
            + [i / j for i, j in zip(total_output_roi.get_shape(), voxel_size)],
        )
        ds.attrs["resolution"] = voxel_size
        ds.attrs["offset"] = total_output_roi.get_offset()

    predict = gp.torch.Predict(
        model,
        checkpoint=model_path,
        inputs={"input": raw},
        outputs={
            0: pred_lsds,
            1: pred_affs,
        },
        spawn_subprocess=True,
    )

    scan = gp.Scan(scan_request, num_workers=20)
    # scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
        dataset_names={
            pred_lsds: out_datasets[0],
            pred_affs: out_datasets[1],
        },
        output_filename=out_file,
    )

    pipeline = (
        source
        # + gp.Pad(raw, context)
        + gp.Normalize(raw)
        + gp.Unsqueeze([raw])
        + gp.Unsqueeze([raw])
        + predict
        + gp.Squeeze([pred_lsds, pred_affs])
        + write
        + scan
    )

    predict_request = gp.BatchRequest()

    predict_request[raw] = total_input_roi
    predict_request[pred_lsds] = total_output_roi
    predict_request[pred_affs] = total_output_roi

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = glob("./model_checkpoint_*")
        model_path.sort(key=os.path.getmtime)
        model_path = os.path.abspath(model_path[-1])

    raw_file = "../../data/xpress-challenge.zarr"
    out_file = "./predictions.zarr"

    raw_dataset = "volumes/training_raw"
    out_datasets = ["pred_lsds", "pred_affs"]
    output_shapes = [10, len(neighborhood)]

    predict(raw_file, raw_dataset, out_file, out_datasets, output_shapes, model_path)
