import gunpowder as gp
import logging
import numpy as np
import os
import glob
import torch
from funlib.learn.torch.models import UNet, ConvPass
from funlib.persistence import prepare_ds
import daisy
import random


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



neighborhood: list[list[int]] = [[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]
def predict_task(
    iteration,
    raw_file,
    raw_dataset,
    out_file="raw_prediction.zarr",
    out_datasets=[(f"pred_affs", len(neighborhood))],
    num_workers=1,
    n_gpu=1,
    model_path="./",
    voxel_size=33,
) -> None:
    if type(iteration) == str and "latest" in iteration:
        model_path = glob.glob(os.path.join(model_path, "model_checkpoint_*"))
        model_path.sort(key=os.path.getmtime)
        model_path = os.path.abspath(model_path[-1])
        print(f"Model path: {model_path}")

    else:
        model_path = os.path.abspath(
            os.path.join(model_path, f"model_checkpoint_{iteration}")
        )

    increase = 8

    input_shape = [132 + increase] * 3
    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5

    # add another downsampling factor for better net (but slower)
    downsample_factors = [(2, 2, 2), (2, 2, 2)]

    unet = UNet(
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        constant_upsample=True,
    )

    model = torch.nn.Sequential(
        unet, ConvPass(num_fmaps, 6, [[1] * 3], activation="Sigmoid")
    )

    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[1:]

    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    voxel_size = gp.Coordinate((voxel_size,) * 3)
    input_size: gp.Coordinate = gp.Coordinate(input_shape) * voxel_size
    output_size: gp.Coordinate = gp.Coordinate(output_shape) * voxel_size

    context: gp.Coordinate = (input_size - output_size) / 2

    raw = gp.ArrayKey("RAW")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    source = gp.ZarrSource(
        raw_file, {raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)
        print(total_output_roi)
    for ds_name, channels in out_datasets:
        logger.info(f"Preparing {ds_name} with {channels} channels...")
        print(voxel_size)
        prepare_ds(
            out_file,
            ds_name,
            total_output_roi,
            voxel_size,
            dtype=np.uint8,
            num_channels=channels,
            write_size=output_size,
            compressor={"id": "blosc"},
            delete=True,
        )

    block_read_roi = daisy.Roi((0,) * 3, input_size) - context
    block_write_roi = daisy.Roi((0,) * 3, output_size)

    def predict():

        in_channels = 1
        num_fmaps = 12
        fmap_inc_factor = 5

        # add another downsampling factor for better net (but slower)
        downsample_factors = [(2, 2, 2), (2, 2, 2)]

        unet = UNet(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            constant_upsample=True,
        )

        model = torch.nn.Sequential(
            unet, ConvPass(num_fmaps, 6, [[1] * 3], activation="Sigmoid")
        )

        model.eval()

        scan_request = gp.BatchRequest()

        scan_request.add(raw, input_size)
        scan_request.add(pred_affs, output_size)

        pred = gp.torch.Predict(
            model,
            checkpoint=model_path,
            inputs={"input": raw},
            outputs={
                0: pred_affs,
            },
        )

        write = gp.ZarrWrite(
            dataset_names={
                pred_affs: out_datasets[0][0],
            },
            output_filename=out_file,
        )

        if num_workers > 1:
            worker_id = int(daisy.Context.from_env()["worker_id"])
            logger.info(worker_id%n_gpu)
            os.environ["CUDA_VISISBLE_DEVICES"] = f"{worker_id % n_gpu}"

            scan = gp.DaisyRequestBlocks(
                scan_request,
                {raw: "read_roi", pred_affs: "write_roi"},
                num_workers=2,
            )

        else:
            scan = gp.Scan(scan_request)

        pipeline = (
            source
            + gp.Normalize(raw)
            + gp.Unsqueeze([raw])
            + gp.Unsqueeze([raw])
            + pred
            + gp.Squeeze([pred_affs])
            + gp.Normalize(pred_affs)
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
            fit="shrink",
        )

        # done: bool = daisy.run_blockwise(tasks=[task])
        done: bool = daisy.run_blockwise([task])

        if not done:
            raise RuntimeError("at least one block failed!")

    else:
        predict()


if __name__ == "__main__":
    iteration = "latest"
    raw_file = "../../data/xpress-challenge.zarr"
    raw_dataset = "volumes/validation_raw"
    out_file = "validation.zarr"
    out_datasets = [
        (f"pred_affs_{iteration}", 6),
    ]

    n_workers = 1
    n_gpu = 1

    predict_task(
        iteration=iteration,
        raw_file="../../data/xpress-challenge.zarr", 
        raw_dataset="volumes/validation_raw", 
        out_file=out_file,
        out_datasets=out_datasets,
        num_workers=n_workers, 
        n_gpu=n_gpu,
        voxel_size=33)
