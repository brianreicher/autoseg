from tqdm import tqdm
import numpy as np
from funlib.persistence import open_ds, prepare_ds, Array
import daisy
import zarr


def correct_blocks(
    raster_file="../../data/xpress-challenge.zarr",
    raster_name="volumes/training_gt_rasters",
    frag_file="./raw_predictions.zarr",
    frag_name="frags_0.1",
    seg_file="./raw_predictions.zarr",
    seg_name="pred_seg",
    num_workers=50,
):

    raster_ds = open_ds(raster_file, raster_name)
    voxel_size = raster_ds.voxel_size
    dtype = raster_ds.dtype
    frag_ds = open_ds(frag_file, frag_name)
    total_roi = frag_ds.roi

    write_roi = daisy.Roi((0,)*3,daisy.Coordinate(frag_ds.chunk_shape))
    read_roi = write_roi.grow(8,8)

    write_roi = write_roi * voxel_size
    read_roi = read_roi * voxel_size

    def worker(block: daisy.Block):
        try:
            raster_ds: Array = open_ds(raster_file, raster_name)
            frag_ds: Array = open_ds(frag_file, frag_name)

            raster_array: np.ndarray = raster_ds.to_ndarray(block.read_roi)
            frag_array: np.ndarray = frag_ds.to_ndarray(block.read_roi)

            frag_array_crop = Array(
                frag_array, block.read_roi, frag_ds.voxel_size
            ).to_ndarray(block.write_roi)

            # Then correct the segmentation
            seg_array: np.ndarray = np.zeros_like(frag_array_crop)
            assert seg_array.shape == frag_array_crop.shape

            for frag_id in tqdm(np.unique(frag_array_crop)):
                seg_ids: list = list(np.unique(raster_array[frag_array == frag_id]))
                if len(seg_ids) == 2:
                    seg_id = [x for x in seg_ids if x != 0][0]
                    seg_array[frag_array_crop == frag_id] = seg_id

            seg_ds = open_ds(seg_file, seg_name, mode="a")
            seg_ds[block.write_roi] = seg_array

            # Now make labels mask
            labels_mask = np.ones_like(seg_array).astype(np.uint8)
            labels_mask_ds = open_ds(seg_file, "pred_labels_mask", mode="a")
            labels_mask_ds[block.write_roi] = labels_mask

            # Now make the unlabelled mask
            unlabelled_mask = (seg_array > 0).astype(np.uint8)
            unlabelled_mask_ds = open_ds(
                seg_file, "pred_unlabelled_mask", mode="a"
            )
            unlabelled_mask_ds[block.write_roi] = unlabelled_mask

            return True
        except Exception as e:
            print("Error in worker")
            print(e)
            return e
            # return False

    ds = prepare_ds(
        seg_file,
        seg_name,
        total_roi=total_roi,
        voxel_size=voxel_size,
        write_size=write_roi.shape,
        dtype=dtype,
        delete=True,
    )

    ds = prepare_ds(
        seg_file,
        "pred_labels_mask",
        total_roi=total_roi,
        voxel_size=voxel_size,
        write_size=write_roi.shape,
        dtype=np.uint8,
        delete=True,
    )

    ds = prepare_ds(
        seg_file,
        "pred_unlabelled_mask",
        total_roi=total_roi,
        voxel_size=voxel_size,
        write_size=write_roi.shape,
        dtype=np.uint8,
        delete=True,
    )

    # create task
    task = daisy.Task(
        "SegCorrectTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=worker,
        num_workers=num_workers,
        max_retries=3,
        fit="shrink"
    )

    # run task
    ret = daisy.run_blockwise([task])
    return ret


if __name__ == "__main__":
    correct_blocks(
        frag_name="frags"
    )
