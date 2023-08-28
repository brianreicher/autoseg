import gunpowder as gp
import logging
import math
import numpy as np
import os
import glob
import sys
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from funlib.persistence import prepare_ds
import daisy

logging.basicConfig(level=logging.INFO)

from model import *


def predict_task(
        iteration, 
        raw_file, 
        raw_dataset, 
        out_file, 
        out_datasets,
        num_workers=1,
        n_gpu=1) -> None:
   
    if "latest" in iteration:
        model_path = glob.glob("./model_checkpoint_*")
        model_path.sort(key=os.path.getmtime)
        model_path = os.path.abspath(model_path[-1])
        print(f"Model path: {model_path}")

    else:
        model_path = os.path.abspath(f"./model_checkpoint_{iteration}")

    # increase = 8 * 15
    # input_shape = [132 + increase] * 3
    # output_shape = model_pre.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[2:]
    # print(input_shape, output_shape)
    # input_shape: list[int] = [252] * 3
    # output_shape: list[int] = [160] * 3
    input_shape: list[int] = [156] * 3
    output_shape: list[int] = [64] * 3

    voxel_size = gp.Coordinate((33,) * 3)
    input_size: gp.Coordinate = gp.Coordinate(input_shape) * voxel_size
    output_size: gp.Coordinate = gp.Coordinate(output_shape) * voxel_size

    context: gp.Coordinate = (input_size - output_size) / 2

    raw = gp.ArrayKey("RAW")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    source = gp.ZarrSource(
        raw_file, {raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)
    for ds_name, channels in out_datasets:
        prepare_ds(
            out_file,
            ds_name,
            total_output_roi,
            voxel_size,
            dtype=np.uint8,
            num_channels=channels,
            write_size=output_size,
            compressor={"id": "blosc"},
            delete=True)

    block_read_roi = daisy.Roi((0,)*3, input_size) - context
    block_write_roi = daisy.Roi((0,)*3, output_size)

    def predict():
        model = MTLSDModel(unet, num_fmaps)
        model.eval()

        scan_request = gp.BatchRequest()

        scan_request.add(raw, input_size)
        scan_request.add(pred_affs, output_size)
        scan_request.add(pred_lsds, output_size)

        pred = gp.torch.Predict(
            model,
            checkpoint=model_path,
            inputs={"input": raw},
            outputs={
                0: pred_lsds,
                1: pred_affs,
            },
        )
        
        write = gp.ZarrWrite(
            dataset_names={
                pred_affs: out_datasets[0][0],
                # pred_lsds: out_datasets[0][1],
            },
            output_filename=out_file,
        )


        if num_workers > 1:
            worker_id = int(daisy.Context.from_env()["worker_id"])
            os.environ["CUDA_VISISBLE_DEVICES"] = f"{worker_id % n_gpu}"

            scan = gp.DaisyRequestBlocks(
                    scan_request, 
                    {
                    raw: "read_roi",
                    pred_lsds: "write_roi",
                    pred_affs: "write_roi"
                    }, 
                    num_workers=1)

        else:
            scan = gp.Scan(scan_request)
        
        pipeline = (
            source
            + gp.Normalize(raw)
            + gp.Unsqueeze([raw])
            + gp.Unsqueeze([raw])
            + pred
            + gp.Squeeze([pred_affs])
            + gp.Normalize(pred_affs, dtype=np.uint8)
            + gp.IntensityScaleShift(pred_affs, 255, 0)
            + write
            + scan
        )

        predict_request = gp.BatchRequest()

        if num_workers == 1:
            predict_request[raw] = total_input_roi
            predict_request[pred_affs] = total_output_roi

        with gp.build(pipeline):
            batch = pipeline.request_batch(predict_request)


    if num_workers > 1:

        task = daisy.Task(
                "PredictBlockwiseTask",
                total_input_roi,
                block_read_roi,
                block_write_roi,
                process_function=predict,
                num_workers=num_workers,
                max_retries=3,
                fit="shrink")

        done: bool = daisy.run_blockwise(tasks=[task])

        if not done:
            raise RuntimeError("at least one block failed!")

    else:
        predict()

if __name__ == "__main__":
    iteration = "latest"
    raw_file = "../../data/xpress-challenge.zarr"
    raw_dataset = "volumes/validation_raw"
    out_file = "validation.zarr"
    out_datasets: list = [(f"pred_affs_{iteration}", 12)]

    n_workers = 1
    n_gpu = 1

    predict_task(
            iteration, 
            raw_file, 
            raw_dataset, 
            out_file, 
            out_datasets,
            n_workers,
            n_gpu)
