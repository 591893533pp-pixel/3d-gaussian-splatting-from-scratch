"""Initialise 3D Gaussian parameters from COLMAP's sparse point cloud."""

from __future__ import annotations

import numpy as np

from .colmap import SparseModel
from .model import GaussianMap


def nearest_neighbor_distances(points: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Returns each point's nearest distinct neighbour distance without SciPy.

    Chunking bounds temporary memory while keeping this reference implementation
    transparent. The later GPU training path will not use this routine.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if len(points) < 2:
        raise ValueError("At least two sparse points are required to initialise scales.")
    result = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        delta = points[start:end, None, :] - points[None, :, :]
        distances_squared = np.einsum("ijk,ijk->ij", delta, delta)
        distances_squared[np.arange(end - start), np.arange(start, end)] = np.inf
        result[start:end] = np.sqrt(np.min(distances_squared, axis=1))
    return result


def initialise_from_sparse_model(model: SparseModel, minimum_scale: float = 1e-4) -> GaussianMap:
    """Creates degree-zero coloured Gaussians with nearest-neighbour scales."""
    if len(model.points) < 2:
        raise ValueError("COLMAP sparse model must contain at least two 3D points.")
    points = list(model.points.values())
    xyz = np.stack([point.xyz for point in points]).astype(np.float32)
    rgb = np.stack([point.rgb for point in points]).astype(np.float32) / 255.0
    radii = np.maximum(nearest_neighbor_distances(xyz), minimum_scale)
    return GaussianMap.from_points(xyz, rgb, radii[:, None])
