from tqdm import tqdm
import numpy as np
import daisy
import zarr
from skimage.morphology import disk, erosion, dilation


def correct_segmentation(
    raster_file="../../data/xpress-challenge.zarr",
    raster_name="volumes/training_gt_rasters",
    frag_file="./prediction.zarr",
    frag_name="frag_seg",
    seg_file="./prediction.zarr",
    seg_name="volumes/pred_seg",
    read_roi_voxels=daisy.Roi((0, 0, 0), (64, 64, 64)),
    write_roi_voxels=daisy.Roi((0, 0, 0), (64, 64, 64)),
    num_workers=20,
    erode_iterations=1,
    erode_footprint=disk(5),
    alternate_dilate=True,
    dilate_footprint=disk(5),
):

    raster_ds = daisy.open_ds(raster_file, raster_name)
    total_roi = raster_ds.roi
    voxel_size = raster_ds.voxel_size
    dtype = raster_ds.dtype
    read_roi = read_roi_voxels * voxel_size
    write_roi = write_roi_voxels * voxel_size

    def worker(block: daisy.Block) -> tuple:
        try:
            raster_ds: daisy.Array = daisy.open_ds(raster_file, raster_name)
            frag_ds: daisy.Array = daisy.open_ds(frag_file, frag_name)

            raster_array: np.ndarray = raster_ds.to_ndarray(block.read_roi)
            frag_array: np.ndarray = frag_ds.to_ndarray(block.read_roi)
            assert raster_array.shape == frag_array.shape

            seg_array: np.ndarray = np.zeros_like(frag_array)

            for frag_id in tqdm(np.unique(frag_array)):
                seg_ids: list = list(np.unique(raster_array[frag_array == frag_id]))
                if len(seg_ids) == 2:
                    seg_ids.pop(0)
                    seg_array[frag_array == frag_id] = seg_ids[0]

            if erode_iterations > 0:
                for _ in range(erode_iterations):
                    seg_array = erosion(seg_array, erode_footprint)
                    if alternate_dilate:
                        seg_array = dilation(seg_array, dilate_footprint)

            seg_ds = daisy.open_ds(seg_file, seg_name)
            seg_ds[block.write_roi] = seg_array
            return True
        except:
            return False

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
        "UpdateSegTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=worker,
        num_workers=num_workers,
    )

    # run task
    ret = daisy.run_blockwise([task])
    return ret


if __name__ == "__main__":
    correct_segmentation()
