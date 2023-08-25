from affogato.segmentation import MWSGridGraph, compute_mws_clustering

import numpy as np
import zarr
from config import *

from typing import Optional


def seeded_mutex_watershed(
    seeds: np.ndarray,
    affs: np.ndarray,
    offsets: "list[tuple[int]]",
    mask: Optional[np.ndarray] = None,
    invert_affinities: bool = False,
    randomize_strides: bool = True,
) -> np.ndarray:
    shape = affs.shape[1:]
    if seeds is not None:
        assert len(seeds.shape) == len(
            shape
        ), f"Got shape {seeds.data.shape} for mask but expected {shape}"
    if mask is not None:
        assert len(mask.shape) == len(
            shape
        ), f"Got shape {mask.data.shape} for mask but expected {shape}"

    grid_graph = MWSGridGraph(shape)
    if seeds is not None:
        grid_graph.update_seeds(seeds.data)

    ndim = len(offsets[0])

    grid_graph.add_attractive_seed_edges = True
    neighbor_affs, lr_affs = (
        np.require(affs[:ndim], requirements="C"),
        np.require(affs[ndim:], requirements="C"),
    )

    # assuming affinities are 1 between voxels that belong together and
    # 0 if they are not part of the same object. Invert if the other way
    # around.
    # neighbors_affs should be high for objects that belong together
    # lr_affs is the oposite
    if invert_affinities:
        neighbor_affs = 1 - neighbor_affs
    else:
        lr_affs = 1 - lr_affs

    uvs, weights = grid_graph.compute_nh_and_weights(neighbor_affs, offsets[:ndim])

    grid_graph.add_attractive_seed_edges = False
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(
        lr_affs,
        offsets[ndim:],
        [4] * ndim,
        randomize_strides=randomize_strides,
    )

    # compute the segmentation
    n_nodes = grid_graph.n_nodes
    segmentation = compute_mws_clustering(
        n_nodes, uvs, mutex_uvs, weights, mutex_weights
    )
    grid_graph.relabel_to_seeds(segmentation)
    segmentation = segmentation.reshape(shape)
    if mask is not None:
        segmentation[np.logical_not(mask)] = 0

    return segmentation


def segment(
    seeds_zarr="../../data/xpress-challenge.zarr",
    seeds_name="volumes/training_gt_rasters",
    pred_zarr="./predictions.zarr",
    affs_name="pred_affs",
    seg_name="pred_seg",
    offsets=neighborhood,
):

    pred_f = zarr.open(pred_zarr, mode="a")
    affs = pred_f[affs_name][:]
    seeds_f = zarr.open(seeds_zarr, mode="r")
    seeds = seeds_f[seeds_name][:]
    segmentation = seeded_mutex_watershed(seeds, affs, offsets)
    ds = pred_f.create_dataset(
        name=seg_name,
        data=segmentation,
        shape=segmentation.shape,
        dtype=segmentation.dtype,
        overwrite=True,
    )
    ds.attrs["resolution"] = pred_f[affs_name].attrs["resolution"]
    ds.attrs["offset"] = pred_f[affs_name].attrs["offset"]


if __name__ == "__main__":
    segment()
