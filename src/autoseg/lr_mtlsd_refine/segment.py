from affogato.segmentation import (
    MWSGridGraph,
    compute_mws_clustering,
    compute_mws_segmentation,
)

import numpy as np
from tqdm import tqdm
import zarr
import daisy
from typing import Optional
import logging
from model import neighborhood
from funlib.persistence import open_ds, prepare_ds, Array
from skimage.morphology import erosion, dilation, ball, remove_small_objects
from scipy.ndimage import distance_transform_edt

from skimage.measure import label


logger = logging.getLogger(__name__)


def expand_labels(labels):
    distance = labels.shape[0]

    distances, indices = distance_transform_edt(labels == 0, return_indices=True)

    expanded_labels = np.zeros_like(labels)

    dilate_mask = distances <= distance

    masked_indices = [dimension_indices[dilate_mask] for dimension_indices in indices]

    nearest_labels = labels[tuple(masked_indices)]

    expanded_labels[dilate_mask] = nearest_labels

    return expanded_labels


def mutex_watershed(
    affs: np.ndarray,
    offsets: "list[tuple[int]]",
    mask_thresh: float = 0.3,
    randomize_strides: bool = True,
    sep: int = 3,
) -> np.ndarray:
    logger.info("Performing mutex watershed...")
    ndim = len(offsets[0])
    # use average affs to mask
    if affs.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs = affs.astype(np.float32)
    else:
        max_affinity_value = 1.0
    mask = np.mean(affs[:sep] / max_affinity_value, axis=0) > mask_thresh

    affs = 1 - affs

    affs[:sep] = affs[:sep] * -1
    affs[:sep] = affs[:sep] + 1

    logger.info("Getting segmentations...")
    segmentation = compute_mws_segmentation(
        affs,
        offsets,
        sep,
        strides=[
            1,
        ]
        * ndim,
        randomize_strides=randomize_strides,
        mask=mask,
    )
    logger.info("Segmented.")

    return segmentation


def seeded_mutex_watershed(
    seeds: np.ndarray,
    affs: np.ndarray,
    offsets: "list[tuple[int]]",
    mask: Optional[np.ndarray] = None,
    invert_affinities: bool = False,
    randomize_strides: bool = True,
    # mask_thresh: float = 0.4,
) -> np.ndarray:
    logger.info("Performing seeded mutex watershed...")
    shape = affs.shape[1:]
    # if affs.dtype == np.uint8:
    #     logger.info("Assuming affinities are in [0,255]")
    #     max_affinity_value = 255.0
    #     affs = affs.astype(np.float32)
    # else:
    #     max_affinity_value = 1.0
    # thresh_mask = np.mean(affs / max_affinity_value, axis=0) > mask_thresh

    if seeds is not None:
        assert len(seeds.shape) == len(
            shape
        ), f"Got shape {seeds.data.shape} for mask but expected {shape}"
    if mask is not None:
        assert len(mask.shape) == len(
            shape
        ), f"Got shape {mask.data.shape} for mask but expected {shape}"

    logger.info("Obtaining grid graph . . .")

    grid_graph = MWSGridGraph(shape)
    if seeds is not None:
        grid_graph.update_seeds(seeds.data)

    ndim = len(offsets[0])

    grid_graph.add_attractive_seed_edges = True
    logger.info("Obtaining lr neighbor affinities")
    neighbor_affs, lr_affs = (
        np.require(affs[:ndim], requirements="C"),
        np.require(affs[ndim:], requirements="C"),
    )
    logger.info("Inverting affs")
    # assuming affinities are 1 between voxels that belong together and
    # 0 if they are not part of the same object. Invert if the other way
    # around.
    # neighbors_affs should be high for objects that belong together
    # lr_affs is the oposite
    if invert_affinities:
        neighbor_affs = 1 - neighbor_affs
    else:
        lr_affs = 1 - lr_affs

    logger.info("Computing weights")
    uvs, weights = grid_graph.compute_nh_and_weights(neighbor_affs, offsets[:ndim])

    grid_graph.add_attractive_seed_edges = False
    logger.info("Computing mutex weights and uvs")
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(
        lr_affs,
        offsets[ndim:],
        [4] * ndim,
        randomize_strides=randomize_strides,
    )

    # compute the segmentation
    n_nodes = grid_graph.n_nodes
    logger.info("Computing segmentation")
    segmentation = compute_mws_clustering(
        n_nodes, uvs, mutex_uvs, weights, mutex_weights
    )
    grid_graph.relabel_to_seeds(segmentation)
    segmentation = segmentation.reshape(shape).astype(np.uint32)

    # mask out the unlabelled area
    #    if seeds is not None:
    #        temp = np.ones_like(segmentation) > 0  # array of True
    #        for label in np.unique(seeds):
    #            temp[segmentation == label] = False
    #        segmentation[temp] = 0

    if mask is not None:
        segmentation[np.logical_not(mask)] = 0

    return segmentation


def segment(
    seeds_zarr="../../data/xpress-challenge.zarr",
    seeds_name="volumes/training_gt_rasters",
    pred_zarr="./validation.zarr",
    affs_name="pred_affs_latest",
    seg_name="pred_seg",
    offsets=neighborhood,
    seeded=False,
) -> None:
    logger.info("Segmenting...")
    logger.info(f"Loading affinities from {pred_zarr}/{affs_name}")
    pred_f = zarr.open(pred_zarr, mode="a")
    affs = pred_f[affs_name][:]

    # run mws segmentation, seeded or unseeded
    if seeded:
        logger.info(f"Loading seeds from {seeds_zarr}/{seeds_name}")
        seeds_f = zarr.open(seeds_zarr, mode="r")
        seeds = seeds_f[seeds_name][:]
        segmentation = seeded_mutex_watershed(seeds, affs, offsets)
    else:
        segmentation = mutex_watershed(affs, offsets)

    logger.info(f"Saving segmentation to {pred_zarr}/{seg_name}")
    ds = pred_f.create_dataset(
        name=seg_name,
        data=segmentation,
        shape=segmentation.shape,
        dtype=segmentation.dtype,
        overwrite=True,
    )
    ds.attrs["resolution"] = pred_f[affs_name].attrs["resolution"]
    ds.attrs["offset"] = pred_f[affs_name].attrs["offset"]

    # Now make labels mask
    labels_mask = np.ones_like(segmentation).astype(np.uint8)
    ds = pred_f.create_dataset(
        name="pred_labels_mask",
        data=labels_mask,
        shape=labels_mask.shape,
        dtype=labels_mask.dtype,
        overwrite=True,
    )
    ds.attrs["resolution"] = pred_f[affs_name].attrs["resolution"]
    ds.attrs["offset"] = pred_f[affs_name].attrs["offset"]

    # Now make the unlabelled mask
    unlabelled_mask = (segmentation > 0).astype(np.uint8)
    ds = pred_f.create_dataset(
        name="pred_unlabelled_mask",
        data=unlabelled_mask,
        shape=unlabelled_mask.shape,
        dtype=unlabelled_mask.dtype,
        overwrite=True,
    )
    ds.attrs["resolution"] = pred_f[affs_name].attrs["resolution"]
    ds.attrs["offset"] = pred_f[affs_name].attrs["offset"]

    logger.info("Done.")


def segment_blocks(
    aff_file="./validation.zarr",
    affs_name="pred_affs_latest",
    frag_file="./valiation.zarr",
    frag_name="frag_seg",
    seg_file="./valiation.zarr",
    seg_name="pred_seg",
    num_workers=20,
    erode_iterations=1,
    erode_footprint=ball(radius=2),
    alternate_dilate=True,
    dilate_footprint=ball(radius=3),
) -> bool:
    offsets = np.array(object=neighborhood)

    aff_ds: Array = open_ds(filename=aff_file, ds_name=affs_name)
    voxel_size = aff_ds.voxel_size
    dtype = aff_ds.dtype
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
        aff_ds: Array = open_ds(aff_file, affs_name)

        affs_array: np.ndarray = aff_ds.to_ndarray(block.read_roi)
        affs_array = (
            affs_array.astype(np.float32) / 255.0
            if affs_array.dtype == np.uint8
            else affs_array
        )

        # First segment the block
        frag_array: np.ndarray = seeded_mutex_watershed(None, affs_array, offsets)

        # # clean up
        # frag_array = remove_small_objects(frag_array, min_size=400).astype(
        #     frag_array.dtype
        # )
        # frag_array = expand_labels(frag_array)
        # frag_array = label(frag_array, connectivity=1).astype(frag_array.dtype)

        # frag_array_crop = Array(
        #     frag_array, block.read_roi, aff_ds.voxel_size
        # ).to_ndarray(block.write_roi)
        # frag_ds: Array = open_ds(frag_file, frag_name, mode="a")
        # frag_ds[block.write_roi] = frag_array_crop

        # # Then correct the segmentation
        # seg_array: np.ndarray = np.zeros_like(frag_array_crop)
        # assert seg_array.shape == frag_array_crop.shape

        # for frag_id in tqdm(np.unique(frag_array_crop)):
        #     seg_ids: list = list(np.unique(raster_array[frag_array == frag_id]))
        #     if len(seg_ids) == 2:
        #         seg_id = [x for x in seg_ids if x != 0][0]
        #         seg_array[frag_array_crop == frag_id] = seg_id

        # if erode_iterations > 0:
        #     for _ in range(erode_iterations):
        #         seg_array = erosion(seg_array, erode_footprint)
        #         if alternate_dilate:
        #             seg_array = dilation(seg_array, dilate_footprint)

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
    segment()
