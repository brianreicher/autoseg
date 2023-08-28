import sys
from predict import predict_task
import rusty_mws
import numpy as np
from model import neighborhood


def get_skel_correct_segmentation(
    predict_affs: bool = True,
    raw_file: str = "../../data/xpress-challenge.zarr",
    raw_dataset: str = "volumes/training_raw",
    out_file: str = "./raw_predictions.zarr",
    out_datasets=[(f"pred_affs", len(neighborhood)), (f"pred_lsds", 10)],
    iteration="latest",
    model_path="./",
    voxel_size:int=100,
) -> None:
    if predict_affs:
        # predict affs
        predict_task(
            iteration=iteration,
            raw_file=raw_file,
            raw_dataset=raw_dataset,
            out_file=out_file,
            out_datasets=out_datasets,
            num_workers=1,
            model_path=model_path,
            voxel_size=voxel_size
        )

    # rusty mws + correction using skeletons
    pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor(
            affs_file=out_file,
            affs_dataset="pred_affs_latest",#out_datasets[0][0],
            fragments_file=out_file,
            fragments_dataset="frag_seg",
            seeds_file=raw_file,
            seeds_dataset="volumes/validation_gt_rasters",
            seg_dataset="pred_seg",
            n_chunk_write_frags=1,
            erode_iterations=1
        )
    
    pp.run_corrected_segmentation_pipeline()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        iteration = sys.argv[1]
        if len(sys.argv) > 2:
            model_path = sys.argv[2]
    else:
        iteration = "latest"
        model_path = "./"

    get_skel_correct_segmentation(iteration=iteration, model_path=model_path, predict_affs=False, out_file="./validation.zarr", voxel_size=33)
