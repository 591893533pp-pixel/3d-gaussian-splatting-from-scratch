"""Reference implementation of 3DGS adaptive density control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import GaussianMap


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


@dataclass
class DensityController:
    gradient_threshold: float = 2e-4
    small_scale: float = 0.05
    min_opacity: float = 0.005

    def __post_init__(self) -> None:
        self.gradient_sum: np.ndarray | None = None
        self.observation_count: np.ndarray | None = None

    def accumulate(self, means_2d_gradient: np.ndarray, visible: np.ndarray) -> None:
        if self.gradient_sum is None or len(self.gradient_sum) != len(means_2d_gradient):
            self.gradient_sum = np.zeros(len(means_2d_gradient), dtype=np.float32)
            self.observation_count = np.zeros(len(means_2d_gradient), dtype=np.int32)
        magnitudes = np.linalg.norm(means_2d_gradient, axis=1)
        self.gradient_sum[visible] += magnitudes[visible]
        self.observation_count[visible] += 1

    def refine(self, gaussian_map: GaussianMap) -> GaussianMap:
        if self.gradient_sum is None or self.observation_count is None:
            return gaussian_map
        average_gradient = self.gradient_sum / np.maximum(self.observation_count, 1)
        scales = gaussian_map.scales
        active = average_gradient >= self.gradient_threshold
        split = active & (np.max(scales, axis=1) > self.small_scale)
        clone = active & ~split
        keep = (gaussian_map.opacity[:, 0] >= self.min_opacity) & ~split
        if not np.any(keep) and not np.any(split):
            keep[np.argmax(gaussian_map.opacity[:, 0])] = True
        means = [gaussian_map.means[keep]]
        log_scales = [gaussian_map.log_scales[keep]]
        quaternions = [gaussian_map.quaternions[keep]]
        opacity_logits = [gaussian_map.opacity_logits[keep]]
        sh = [gaussian_map.sh_coefficients[keep]]
        for index in np.flatnonzero(clone):
            means.append(gaussian_map.means[index:index + 1])
            log_scales.append(gaussian_map.log_scales[index:index + 1])
            quaternions.append(gaussian_map.quaternions[index:index + 1])
            opacity_logits.append(gaussian_map.opacity_logits[index:index + 1])
            sh.append(gaussian_map.sh_coefficients[index:index + 1])
        for index in np.flatnonzero(split):
            rotation = _quaternion_matrix(gaussian_map.quaternions[index])
            largest_axis = int(np.argmax(scales[index]))
            offset = rotation[:, largest_axis] * scales[index, largest_axis] * 0.5
            means.append(np.stack((gaussian_map.means[index] - offset, gaussian_map.means[index] + offset)))
            log_scales.append(np.repeat((gaussian_map.log_scales[index] - np.log(1.6))[None, :], 2, axis=0))
            quaternions.append(np.repeat(gaussian_map.quaternions[index:index + 1], 2, axis=0))
            opacity_logits.append(np.repeat(gaussian_map.opacity_logits[index:index + 1], 2, axis=0))
            sh.append(np.repeat(gaussian_map.sh_coefficients[index:index + 1], 2, axis=0))
        self.gradient_sum = None
        self.observation_count = None
        return GaussianMap(np.concatenate(means), np.concatenate(log_scales), np.concatenate(quaternions), np.concatenate(opacity_logits), np.concatenate(sh))
