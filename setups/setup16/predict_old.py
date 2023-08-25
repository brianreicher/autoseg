from glob import glob
import logging
import math
import numpy as np
import os
import sys
import zarr
import daisy
from config import *

logging.basicConfig(level=logging.INFO)
# os.environ[
#     "DAISY_CONTEXT"
# ] = f"hostname={os.environ['HOSTNAME']}:port=34313:task_id=PredictTask:worker_id=0"

increase *= 5

input_shape = [132 + increase] * 3
output_shape = model.forward(torch.empty(size=[1, 1] + input_shape)).shape[-3:]

input_size = gp.Coordinate(input_shape) * voxel_size
output_size = gp.Coordinate(output_shape) * voxel_size

context = (input_size - output_size) // 2


def predict(
    raw_file,
    raw_dataset,
    raw_shape=1,
    out_file="predictions.zarr",
    out_datasets=["pred_lsds", "pred_affs"],
    out_shapes=[10, len(neighborhood)],
    model=model,
    model_path="./latest",
    input_size=input_size,
    output_size=output_size,
    context=context,
):
    if "latest" in model_path:
        model_path = glob(f"{model_path.replace('latest', '')}model_checkpoint_*")
        model_path.sort(key=os.path.getmtime)
        model_path = os.path.abspath(model_path[-1])

    raw = gp.ArrayKey("RAW")
    out_keys = [gp.ArrayKey(out_dataset.upper()) for out_dataset in out_datasets]

    # pred_lsds = gp.ArrayKey("PRED_LSDS")
    # pred_affs = gp.ArrayKey("PRED_AFFS")

    model.eval()

    source = gp.ZarrSource(
        raw_file, {raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    outputs = {}
    roi_map = {raw: "read_roi"}
    dataset_names = {}
    # predict_request = gp.BatchRequest()
    # predict_request[raw] = total_input_roi
    # predict_request[pred_lsds] = total_output_roi
    # predict_request[pred_affs] = total_output_roi
    for i, out_key in enumerate(out_keys):
        outputs[i] = out_key
        roi_map[out_key] = "write_roi"
        dataset_names[out_key] = out_datasets[i]
        scan_request.add(out_key, output_size)
        # predict_request[out_key] = total_output_roi

    # scan_request.add(pred_lsds, output_size)
    # scan_request.add(pred_affs, output_size)

    f = zarr.open(out_file, "a")
    for out_dataset, output_shape in zip(out_datasets, out_shapes):
        ds = f.create_dataset(
            out_dataset,
            shape=[output_shape]
            + [i / j for i, j in zip(total_output_roi.get_shape(), voxel_size)],
            overwrite=True,
        )
        ds.attrs["resolution"] = voxel_size
        ds.attrs["offset"] = total_output_roi.get_offset()

    predict = gp.torch.Predict(
        model,
        checkpoint=model_path,
        inputs={"input": raw},
        outputs=outputs,
        spawn_subprocess=True,
    )

    # scan = gp.DaisyRequestBlocks(scan_request, roi_map, num_workers=20)
    # scan = gp.Scan(scan_request, num_workers=20)
    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
        dataset_names=dataset_names,
        output_filename=out_file,
    )

    unsqueeze = gp.Unsqueeze([raw])
    if raw_shape == 1:
        unsqueeze += gp.Unsqueeze([raw])

    pipeline = (
        source
        + gp.Pad(raw, context)
        + gp.Normalize(raw)
        + unsqueeze
        + predict
        + gp.Squeeze(out_keys)
        + write
        + scan
    )

    with gp.build(pipeline):
        # pipeline.request_batch(predict_request)
        pipeline.request_batch(gp.BatchRequest())


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
