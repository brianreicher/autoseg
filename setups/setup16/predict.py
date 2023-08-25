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
#    "DAISY_CONTEXT"
# ] = f"hostname={os.environ['HOSTNAME']}:port=34313:task_id=PredictTask:worker_id=0"

increase *= 5

input_shape = [132 + increase] * 3
# output_shape = model.forward(torch.empty(size=[1, 1] + input_shape)).shape[-3:]
# print(output_shape)
output_shape = [160] * 3

input_size = gp.Coordinate(input_shape) * voxel_size
output_size = gp.Coordinate(output_shape) * voxel_size

context = (input_size - output_size) // 2


def predict_task(
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
    num_workers=1,
    n_gpu=1,
    ndims=3,
    roi=None,
):

    if "latest" in model_path:
        model_path = glob(f"{model_path.replace('latest', '')}model_checkpoint_*")
        model_path.sort(key=os.path.getmtime)
        model_path = os.path.abspath(model_path[-1])

    source = daisy.open_ds(raw_file, raw_dataset)
    if roi is None:
        input_roi = source.roi  # .grow(context, context)
        output_roi = source.roi.grow(-context, -context)
    else:
        input_roi = roi.grow(context, context)
        output_roi = roi

    for out_dataset, output_shape in zip(out_datasets, out_shapes):

        daisy.prepare_ds(
            out_file,
            out_dataset,
            output_roi,
            source.voxel_size,
            dtype=np.uint8,
            num_channels=output_shape,
            compressor={"id": "blosc"},
            delete=True,
        )

    block_read_roi = daisy.Roi((0,) * ndims, input_size) - context
    block_write_roi = daisy.Roi((0,) * ndims, output_size)

    model.eval()

    def predict():

        raw = gp.ArrayKey("RAW")
        out_keys = [gp.ArrayKey(out_dataset.upper()) for out_dataset in out_datasets]

        source = gp.ZarrSource(
            raw_file, {raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True)}
        )

        scan_request = gp.BatchRequest()

        scan_request.add(raw, input_size)
        outputs = {}
        roi_map = {raw: "read_roi"}
        dataset_names = {}

        for i, out_key in enumerate(out_keys):
            outputs[i] = out_key
            roi_map[out_key] = "write_roi"
            dataset_names[out_key] = out_datasets[i]
            scan_request.add(out_key, output_size)

        predict = gp.torch.Predict(
            model,
            checkpoint=model_path,
            inputs={"input": raw},
            outputs=outputs,
            spawn_subprocess=True,
        )

        for out_key in out_keys:
            predict += gp.IntensityScaleShift(out_key, 255, 0)
            predict += gp.Normalize(out_key, factor=1, dtype=np.uint8)

        if num_workers > 1:
            worker_id = int(daisy.Context.from_env()["worker_id"])
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{worker_id % n_gpu}"

            scan = gp.DaisyRequestBlocks(scan_request, roi_map, num_workers=1)
        else:
            # scan = gp.Scan(scan_request, num_workers=num_workers)
            scan = gp.Scan(scan_request, num_workers=1)

        write = gp.ZarrWrite(
            dataset_names=dataset_names,
            output_filename=out_file,
        )

        unsqueeze = gp.Unsqueeze([raw])
        if raw_shape == 1:
            unsqueeze += gp.Unsqueeze([raw])
            unsqueeze += gp.IntensityScaleShift(raw, 2, -1)

        pipeline = (
            source
            + gp.Pad(raw, None)
            + gp.Normalize(raw)
            + unsqueeze
            + predict
            + gp.Squeeze(out_keys)
            + write
            + scan
        )

        predict_request = gp.BatchRequest()

        if num_workers == 1:
            predict_request[raw] = input_roi
            for out_key in out_keys:
                predict_request[out_key] = output_roi

        with gp.build(pipeline):
            pipeline.request_batch(predict_request)

    if num_workers > 1:
        # process block-wise
        task = daisy.Task(
            "PredictBlockwiseTask",
            input_roi,
            block_read_roi,
            block_write_roi,
            process_function=predict,
            num_workers=num_workers,
            read_write_conflict=False,
            max_retries=5,
            fit="shrink",
        )

        done = daisy.run_blockwise([task])

        if not done:
            raise RuntimeError("at least one block failed!")
    else:
        predict()


if __name__ == "__main__":
    #    if len(sys.argv) > 1:
    #        model_path = sys.argv[1]
    #    else:
    #        model_path = glob("./model_checkpoint_*")
    #        model_path.sort(key=os.path.getmtime)
    #        model_path = os.path.abspath(model_path[-1])

    raw_file = "../../data/xpress-challenge.zarr"
    out_file = "./predictions.zarr"

    predict_task(
        raw_file=raw_file,
        raw_dataset="volumes/training_raw",
        out_file=out_file,
        out_datasets=["pred_lsds"],
        out_shapes=[10],
        model_path="./model_checkpoint_3000",
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
