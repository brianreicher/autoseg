from config import *
from predict import predict_task
from segment_correct_blocks import segment_correct_blocks
from unseeded_segment_blocks import unseeded_segment_blocks
from unseeded_segment import unseeded_segment
import daisy


def get_segmentation(
    raw_file="../../data/xpress-challenge.zarr",
    raw_dataset="volumes/training_raw",
    out_file="./predictions.zarr",
    model_path: str = "./latest",
    # aff_model_path="../setup15.1/latest",
    aff_model_path="../setup15.1/model_checkpoint_4000",
):
    predict_task(
        raw_file=raw_file,
        raw_dataset=raw_dataset,
        out_file=out_file,
        out_datasets=["pred_lsds"],
        out_shapes=[10],
        model_path=model_path,
    )
    predict_task(
        raw_file=out_file,
        raw_dataset="pred_lsds",
        raw_shape=10,
        out_file=out_file,
        out_datasets=["pred_affs"],
        out_shapes=[len(neighborhood)],
        model=aff_model,
        model_path=aff_model_path,
        input_size=aff_input_size,
        output_size=aff_output_size,
        context=aff_context,
    )
    # segment()
    # correct_segmentation()
    return segment_correct_blocks()


def get_validation_segmentation(
    raw_file="../../data/xpress-challenge.zarr",
    raw_dataset="volumes/validation_raw",
    out_file="./validation.zarr",
    model_path: str = "./model_checkpoint_30000",
    # model_path: str = "./latest",
    # aff_model_path="../setup15.1/latest",
    aff_model_path="../setup15.1/model_checkpoint_4000",
):
    predict_task(
        raw_file=raw_file,
        raw_dataset=raw_dataset,
        out_file=out_file,
        out_datasets=["pred_lsds"],
        out_shapes=[10],
        model_path=model_path,
    )
    predict_task(  # LSD --> Affinities
        raw_file=out_file,
        raw_dataset="pred_lsds",
        raw_shape=10,
        out_file=out_file,
        out_datasets=["pred_affs"],
        out_shapes=[len(neighborhood)],
        model=aff_model,
        model_path=aff_model_path,
        input_size=aff_input_size,
        output_size=aff_output_size,
        context=aff_context,
    )
    return unseeded_segment(
        aff_file=out_file,
        seg_file="./validation.zarr",
        seg_name="validation",
        downsample=3,
    )


def get_test_segmentation(
    raw_file="../../data/xpress-challenge.zarr",
    raw_dataset="volumes/test_raw",
    out_file="./test.zarr",
    model_path: str = "./model_checkpoint_30000",
    # model_path: str = "./latest",
    # aff_model_path="../setup15.1/latest",
    aff_model_path="../setup15.1/model_checkpoint_4000",
):
    roi_begin = "3267,3267,3267"
    roi_shape = "33066,33066,33066"

    # Parse ROI
    roi = None
    if roi_begin is not None or roi_begin is not None:
        roi_begin = [float(k) for k in roi_begin.split(",")]
        roi_shape = [float(k) for k in roi_shape.split(",")]
        roi = daisy.Roi(roi_begin, roi_shape)

    predict_task(
        raw_file=raw_file,
        raw_dataset=raw_dataset,
        out_file=out_file,
        out_datasets=["pred_lsds"],
        out_shapes=[10],
        model_path=model_path,
        roi=roi.grow(context, context),
    )
    predict_task(  # LSD --> Affinities
        raw_file=out_file,
        raw_dataset="pred_lsds",
        raw_shape=10,
        out_file=out_file,
        out_datasets=["pred_affs"],
        out_shapes=[len(neighborhood)],
        model=aff_model,
        model_path=aff_model_path,
        input_size=aff_input_size,
        output_size=aff_output_size,
        context=aff_context,
        roi=roi,
    )
    return unseeded_segment(
        aff_file=out_file,
        seg_file="./submission.zarr",
        seg_name="submission",
        downsample=3,
    )


if __name__ == "__main__":
    # get_validation_segmentation(model_path="./model_checkpoint_30000")
    # get_segmentation(model_path="./model_checkpoint_30000")
    # get_segmentation()
    get_test_segmentation(model_path="./model_checkpoint_30000")
