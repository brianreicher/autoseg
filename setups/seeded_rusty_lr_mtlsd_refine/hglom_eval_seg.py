from predict import predict_task
from model import neighborhood
import heiarchical_agglom as hglom


def get_validation_segmentation(
    iteration="latest",
    raw_file="../../data/xpress-challenge.zarr",
    raw_dataset="volumes/validation_raw",
    out_file="./validation.zarr",
    pred_affs=True
) -> bool:

    affs_ds: str = f"pred_affs_{iteration}"

    if pred_affs:
        predict_task( # Raw --> Affinities
            iteration=iteration,
            raw_file=raw_file,
            raw_dataset=raw_dataset,
            out_file=out_file,
            out_datasets=[(affs_ds, len(neighborhood)), ("pred_lsds_latest", 10)],
            num_workers=1,
            n_gpu=1)

    pp: hglom.PostProcessor = hglom.PostProcessor( # Affinities -> Segmentation
                affs_file=out_file,
                affs_dataset=affs_ds,
                seeds_file=raw_file,
                seeds_dataset="volumes/validation_gt_rasters",
                seg_dataset="pred_seg_hgolm",
                neighborhood_length=len(neighborhood),
                filter_val=.46)
    
    return pp.run_hierarchical_agglom_segmentation_pipeline()


if __name__ == "__main__":
    get_validation_segmentation(raw_file="../../data/xpress-challenge.zarr",
                                raw_dataset="volumes/validation_raw",
                                out_file="./validation.zarr",
                                pred_affs=True)