from tqdm import tqdm
import numpy as np
from funlib.persistence import open_ds, prepare_ds, Array
import daisy
import zarr
from skimage.morphology import disk, erosion, dilation, ball, remove_small_objects
from skimage.segmentation import expand_labels as _expand_labels
from scipy.ndimage import distance_transform_edt
from skimage.measure import label

from segment import seeded_mutex_watershed
from model import neighborhood


def expand_labels(labels):
    distance = labels.shape[0]

    distances, indices = distance_transform_edt(labels == 0, return_indices=True)

    expanded_labels = np.zeros_like(labels)

    dilate_mask = distances <= distance

    masked_indices = [dimension_indices[dilate_mask] for dimension_indices in indices]

    nearest_labels = labels[tuple(masked_indices)]

    expanded_labels[dilate_mask] = nearest_labels

    return expanded_labels


def segment_correct_blocks(
    raster_file="../../data/xpress-challenge.zarr",
    raster_name="volumes/training_gt_rasters",
    aff_file="./raw_predictions.zarr",
    affs_name="pred_affs_latest",
    frag_file="./raw_predictions.zarr",
    frag_name="frag_seg",
    seg_file="./raw_predictions.zarr",
    seg_name="pred_seg",
    num_workers=20,
    erode_iterations=1,
    erode_footprint=ball(radius=2),
    alternate_dilate=True,
    dilate_footprint=ball(radius=3),
) -> bool:
    offsets = np.array(object=neighborhood)

    raster_ds = open_ds(filename=raster_file, ds_name=raster_name)
    voxel_size = raster_ds.voxel_size
    dtype = raster_ds.dtype
    aff_ds: Array = open_ds(filename=aff_file, ds_name=affs_name)
    total_roi = aff_ds.roi

    write_roi = daisy.Roi(
        offset=(0,) * 3, shape=daisy.Coordinate(aff_ds.chunk_shape)[1:]
    )

    min_neighborhood: int = min(
        filter(
            lambda x: x != 0, [value for sublist in neighborhood for value in sublist]
        )
    )
    max_neighborhood: int = max(
        filter(
            lambda x: x != 0, [value for sublist in neighborhood for value in sublist]
        )
    )

    read_roi = write_roi.grow(amount_neg=min_neighborhood, amount_pos=max_neighborhood)

    write_roi = write_roi * voxel_size
    read_roi = read_roi * voxel_size

    def worker(block: daisy.Block):
        try:
            raster_ds: Array = open_ds(raster_file, raster_name)
            aff_ds: Array = open_ds(aff_file, affs_name)

            raster_array: np.ndarray = raster_ds.to_ndarray(block.read_roi)
            affs_array: np.ndarray = aff_ds.to_ndarray(block.read_roi)
            affs_array = (
                affs_array.astype(np.float32) / 255.0
                if affs_array.dtype == np.uint8
                else affs_array
            )

            # First segment the block
            frag_array: np.ndarray = seeded_mutex_watershed(None, affs_array, offsets)

            # clean up
            frag_array = remove_small_objects(frag_array, min_size=400).astype(
                frag_array.dtype
            )
            frag_array = expand_labels(frag_array)
            frag_array = label(frag_array, connectivity=1).astype(frag_array.dtype)

            frag_array_crop = Array(
                frag_array, block.read_roi, aff_ds.voxel_size
            ).to_ndarray(block.write_roi)
            frag_ds: Array = open_ds(frag_file, frag_name, mode="a")
            frag_ds[block.write_roi] = frag_array_crop

            # Then correct the segmentation
            seg_array: np.ndarray = np.zeros_like(frag_array_crop)
            assert seg_array.shape == frag_array_crop.shape

            for frag_id in tqdm(np.unique(frag_array_crop)):
                seg_ids: list = list(np.unique(raster_array[frag_array == frag_id]))
                if len(seg_ids) == 2:
                    seg_id = [x for x in seg_ids if x != 0][0]
                    seg_array[frag_array_crop == frag_id] = seg_id

            if erode_iterations > 0:
                for _ in range(erode_iterations):
                    seg_array = erosion(seg_array, erode_footprint)
                    if alternate_dilate:
                        seg_array = dilation(seg_array, dilate_footprint)

            seg_ds: Array = open_ds(seg_file, seg_name, mode="a")
            seg_ds[block.write_roi] = seg_array

            # Now make labels mask
            labels_mask = np.ones_like(seg_array).astype(np.uint8)
            labels_mask_ds = open_ds(seg_file, "pred_labels_mask", mode="a")
            labels_mask_ds[block.write_roi] = labels_mask

            # Now make the unlabelled mask
            unlabelled_mask = (seg_array > 0).astype(np.uint8)
            unlabelled_mask_ds = open_ds(seg_file, "pred_unlabelled_mask", mode="a")
            unlabelled_mask_ds[block.write_roi] = unlabelled_mask

            return True
        except Exception as e:
            print("Error in worker")
            print(e)
            return e
            # return False

    ds = prepare_ds(
        frag_file,
        frag_name,
        total_roi=total_roi,
        voxel_size=voxel_size,
        write_size=write_roi.shape,
        dtype=dtype,
        delete=True,
    )

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
        ds_name="pred_unlabelled_mask",
        total_roi=total_roi,
        voxel_size=voxel_size,
        write_size=write_roi.shape,
        dtype=np.uint8,
        delete=True,
    )

    # create task
    task = daisy.Task(
        task_id="SegCorrectTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=worker,
        num_workers=num_workers,
        fit="shrink",
        max_retries=3,
    )

    # run task
    ret: bool = daisy.run_blockwise(tasks=[task])
    return ret


if __name__ == "__main__":
    segment_correct_blocks()
