from tqdm import tqdm
import numpy as np
import daisy
import zarr
from skimage.morphology import disk, erosion, dilation, ball, remove_small_objects
from funlib.segment.arrays import relabel, replace_values

from scipy.ndimage import measurements

import logging

logger = logging.getLogger(__name__)

from fragments import watershed_from_affinities
from segment import mutex_watershed
from config import *


def filtered_segment(
    aff_file="./validation.zarr",
    affs_name="pred_affs",
    seg_file="./validation.zarr",
    seg_name="validation",
    read_roi_voxels=daisy.Roi((0, 0, 0), (128, 128, 128)).grow(
        -neighborhood.min(), neighborhood.max()
    ),
    write_roi_voxels=daisy.Roi((0, 0, 0), (128, 128, 128)),
    num_workers=32,
    erode_iterations=0,
    erode_footprint=ball(6),
    alternate_dilate=True,
    dilate_footprint=ball(5),
    offsets=neighborhood,
    size_threshold=None,  # 100,
    downsample=1,
    filter_fragments=0.05,
    roi=None,
):

    dtype = np.uint32
    aff_ds = daisy.open_ds(aff_file, affs_name)
    voxel_size = aff_ds.voxel_size
    if roi is None:
        total_roi = aff_ds.roi
    else:
        total_roi = roi

    logger.info("Loading affs...")
    affs_array: np.ndarray = aff_ds.to_ndarray(total_roi)
    if downsample > 1:
        logger.info("Downsampling...")
        affs_array = affs_array[..., ::downsample, ::downsample, ::downsample]
        voxel_size *= downsample
        roi_begin = (aff_ds.roi.begin // voxel_size) * voxel_size
        roi_shape = daisy.Coordinate(affs_array.shape[-3:]) * voxel_size
        total_roi = daisy.Roi(roi_begin, roi_shape)

    read_roi = read_roi_voxels * voxel_size
    write_roi = write_roi_voxels * voxel_size

    if affs_array.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs_array = affs_array.astype(np.float32)
    else:
        max_affinity_value = 1.0

    # extract fragments
    fragments_data, _ = watershed_from_affinities(
        affs_array,
        max_affinity_value,
        # min_seed_distance=min_seed_distance,
    )

    average_affs = np.mean(affs_array / max_affinity_value, axis=0)

    filtered_fragments = []

    fragment_ids = np.unique(fragments_data)

    for fragment, mean in zip(
        fragment_ids, measurements.mean(average_affs, fragments_data, fragment_ids)
    ):
        if mean < filter_fragments:
            filtered_fragments.append(fragment)

    filtered_fragments = np.array(filtered_fragments, dtype=fragments_data.dtype)
    replace = np.zeros_like(filtered_fragments)
    replace_values(fragments_data, filtered_fragments, replace, inplace=True)

    affs_array[:, fragments_data > 0] = 0

    # First segment the block
    logger.info("Segmenting...")
    seg_array: np.ndarray = mutex_watershed(affs_array, offsets)

    if erode_iterations > 0:
        logger.info("Eroding...")
        for _ in range(erode_iterations):
            seg_array = erosion(seg_array, erode_footprint)
            if alternate_dilate:
                seg_array = dilation(seg_array, dilate_footprint)

    if size_threshold is not None and size_threshold > 0:
        logger.info("Removing small objects...")
        seg_array = np.absolute(seg_array, dtype=np.int64)
        seg_array = remove_small_objects(seg_array, size_threshold)

    seg_array = daisy.Array(seg_array, total_roi, voxel_size)

    def worker(block: daisy.Block):
        try:
            seg_ds = daisy.open_ds(seg_file, seg_name, mode="a")
            seg_ds[block.write_roi] = seg_array.to_ndarray(block.write_roi)

            return True
        except Exception as e:
            logger.info("Error in worker")
            logger.info(e)
            return e
            # return False

    logger.info(f"Writing {seg_name} to {seg_file}...")
    ds = daisy.prepare_ds(
        seg_file,
        seg_name,
        total_roi=total_roi,
        voxel_size=voxel_size,
        write_size=write_roi.shape,
        dtype=dtype,
        delete=True,
    )

    # create task
    task = daisy.Task(
        "SegTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=worker,
        num_workers=num_workers,
        max_retries=3,
    )

    # run task
    ret = daisy.run_blockwise([task])
    return ret


if __name__ == "__main__":
    raw_file = "../../data/xpress-challenge.zarr"
    raw_dataset = "volumes/test_raw"
    out_file = "./test.zarr"
    model_path: str = "./latest"
    aff_model_path = "../setup15.1/model_checkpoint_4000"
    roi_begin = "3267,3267,3267"
    roi_shape = "33066,33066,33066"

    # Parse ROI
    roi = None
    if roi_begin is not None or roi_begin is not None:
        roi_begin = [float(k) for k in roi_begin.split(",")]
        roi_shape = [float(k) for k in roi_shape.split(",")]
        roi = daisy.Roi(roi_begin, roi_shape)

    filtered_segment(
        aff_file=out_file,
        seg_file="./submission.zarr",
        seg_name="submission_filtered",
        # downsample=3,
        roi=roi,
    )
