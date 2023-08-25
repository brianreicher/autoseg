from tqdm import tqdm
import numpy as np
import daisy
import zarr
from skimage.morphology import disk, erosion, dilation, ball, remove_small_objects

from segment import mutex_watershed
from config import *


def unseeded_segment_blocks(
    aff_file="./predictions.zarr",
    affs_name="pred_affs",
    seg_file="./predictions.zarr",
    seg_name="pred_seg",
    # read_roi_voxels=daisy.Roi((0, 0, 0), (128, 128, 128)).grow(
    #     -neighborhood.min(), neighborhood.max()
    # ),
    write_roi_voxels=daisy.Roi((0, 0, 0), (128, 128, 128)),
    num_workers=32,
    erode_iterations=1,
    erode_footprint=ball(6),
    alternate_dilate=True,
    dilate_footprint=ball(5),
    offsets=neighborhood,
    make_masks=False,
):

    dtype = np.uint32
    aff_ds = daisy.open_ds(aff_file, affs_name)
    voxel_size = aff_ds.voxel_size
    total_roi = aff_ds.roi
    # read_roi = read_roi_voxels * voxel_size
    write_roi = write_roi_voxels * voxel_size

    def worker(block: daisy.Block):
        try:
            aff_ds: daisy.Array = daisy.open_ds(aff_file, affs_name)
            affs_array: np.ndarray = aff_ds.to_ndarray(block.read_roi)
            # First segment the block
            seg_array: np.ndarray = mutex_watershed(affs_array, offsets)

            if erode_iterations > 0:
                for _ in range(erode_iterations):
                    seg_array = erosion(seg_array, erode_footprint)
                    if alternate_dilate:
                        seg_array = dilation(seg_array, dilate_footprint)

            if not (block.read_roi == block.write_roi):
                seg_array = daisy.Array(
                    seg_array, block.read_roi, aff_ds.voxel_size
                ).to_ndarray(block.write_roi)
            seg_array = remove_small_objects(seg_array, 100)

            seg_ds = daisy.open_ds(seg_file, seg_name, mode="a")
            seg_ds[block.write_roi] = seg_array

            if make_masks:
                # Now make labels mask
                labels_mask = np.ones_like(seg_array).astype(np.uint8)
                labels_mask_ds = daisy.open_ds(seg_file, "pred_labels_mask", mode="a")
                labels_mask_ds[block.write_roi] = labels_mask

                # Now make the unlabelled mask
                unlabelled_mask = (seg_array > 0).astype(np.uint8)
                unlabelled_mask_ds = daisy.open_ds(
                    seg_file, "pred_unlabelled_mask", mode="a"
                )
                unlabelled_mask_ds[block.write_roi] = unlabelled_mask

            return True
        except Exception as e:
            print("Error in worker")
            print(e)
            return e
            # return False

    ds = daisy.prepare_ds(
        seg_file,
        seg_name,
        total_roi=total_roi,
        voxel_size=voxel_size,
        write_size=write_roi.shape,
        dtype=dtype,
        delete=True,
    )

    if make_masks:
        ds = daisy.prepare_ds(
            seg_file,
            "pred_labels_mask",
            total_roi=total_roi,
            voxel_size=voxel_size,
            write_size=write_roi.shape,
            dtype=np.uint8,
            delete=True,
        )

        ds = daisy.prepare_ds(
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
        "SegTask",
        total_roi=total_roi,
        # read_roi=read_roi,
        # write_roi=write_roi,
        read_roi=total_roi,
        write_roi=total_roi,
        process_function=worker,
        num_workers=num_workers,
        max_retries=3,
    )

    # run task
    ret = daisy.run_blockwise([task])
    return ret


if __name__ == "__main__":
    unseeded_segment_blocks()
