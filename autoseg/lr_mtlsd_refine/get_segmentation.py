from predict import predict_task
from segment_correct_blocks import segment_correct_blocks


def get_validation_segmentation(
    iteration="latest",
    raw_file="../../data/xpress-challenge.zarr",
    raw_dataset="volumes/validation_raw",
    out_file="./validation.zarr",
    pred_affs:bool=True,
) -> bool:

    affs_ds: str = f"pred_affs_{iteration}"

    if pred_affs:
        predict_task( # Raw --> Affinities
            iteration=iteration,
            raw_file=raw_file,
            raw_dataset=raw_dataset,
            out_file=out_file,
            out_datasets=[(affs_ds, 12)],
            num_workers=1,
            n_gpu=1)
    
    segment_correct_blocks(raster_file="../../data/xpress-challenge.zarr",
                            raster_name="volumes/validation_gt_rasters",
                            aff_file="./validation.zarr",
                            affs_name="pred_affs_latest",
                            frag_file="./validation.zarr",
                            frag_name="frag_seg",
                            seg_file="./validation.zarr",
                            seg_name="pred_seg")

    return True


if __name__ == "__main__":
    get_validation_segmentation(pred_affs=False)
