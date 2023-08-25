from predict import predict_task
import rusty_mws


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
            out_datasets=[(affs_ds, 3)],
            num_workers=1,
            n_gpu=1)

    pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor( # Affinities -> Segmentation
                affs_file=out_file,
                affs_dataset=affs_ds,
                seg_dataset="pred_seg_-1.5_1.5_0.25",
                neighborhood_length=3,
                n_chunk_write_frags=1,
                filter_val=.25,
                lr_bias=-1.5,
                adj_bias=1.5
            )
    
    return pp.run_pred_segmentation_pipeline()


if __name__ == "__main__":
    get_validation_segmentation(raw_file="../../data/xpress-challenge.zarr",
                                raw_dataset="volumes/validation_raw",
                                out_file="./validation.zarr",
                                pred_affs=True)