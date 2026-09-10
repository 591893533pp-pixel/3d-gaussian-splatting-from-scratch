"""Core Gaussian-map representation and portable serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _normalise_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("A quaternion must have non-zero length.")
    return quaternions / norms


@dataclass
class GaussianMap:
    """Trainable parameters for a set of anisotropic 3D Gaussians.

    Scales are stored in log space and opacity as logits, matching the stable
    parameterisation used by the original 3DGS optimisation.
    """

    means: np.ndarray
    log_scales: np.ndarray
    quaternions: np.ndarray
    opacity_logits: np.ndarray
    sh_coefficients: np.ndarray

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32)
        self.log_scales = np.asarray(self.log_scales, dtype=np.float32)
        self.quaternions = np.asarray(self.quaternions, dtype=np.float32)
        self.opacity_logits = np.asarray(self.opacity_logits, dtype=np.float32).reshape(-1, 1)
        self.sh_coefficients = np.asarray(self.sh_coefficients, dtype=np.float32)
        self._validate()
        self.quaternions = _normalise_quaternions(self.quaternions).astype(np.float32)

    def _validate(self) -> None:
        count = self.means.shape[0]
        if self.means.ndim != 2 or self.means.shape[1] != 3:
            raise ValueError("means must have shape (N, 3).")
        if self.log_scales.shape != (count, 3):
            raise ValueError("log_scales must have shape (N, 3).")
        if self.quaternions.shape != (count, 4):
            raise ValueError("quaternions must have shape (N, 4), in wxyz order.")
        if self.opacity_logits.shape != (count, 1):
            raise ValueError("opacity_logits must have shape (N, 1).")
        if self.sh_coefficients.ndim != 3 or self.sh_coefficients.shape[:2] != (count, 3):
            raise ValueError("sh_coefficients must have shape (N, 3, K).")

    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def scales(self) -> np.ndarray:
        return np.exp(self.log_scales)

    @property
    def opacity(self) -> np.ndarray:
        return _sigmoid(self.opacity_logits)

    @property
    def base_colors(self) -> np.ndarray:
        """Returns the degree-zero SH colour in displayable RGB space."""
        return np.clip(0.5 + SH_C0 * self.sh_coefficients[:, :, 0], 0.0, 1.0)

    @classmethod
    def from_points(cls, points: np.ndarray, colors: np.ndarray, scale: float | np.ndarray = 0.03) -> "GaussianMap":
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
            raise ValueError("points and colors must both have shape (N, 3).")
        count = len(points)
        scales = np.broadcast_to(np.asarray(scale, dtype=np.float32), (count, 3))
        if np.any(scales <= 0):
            raise ValueError("scale must be strictly positive.")
        return cls(
            means=points,
            log_scales=np.log(scales).astype(np.float32),
            quaternions=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1)),
            opacity_logits=np.full((count, 1), 2.0, dtype=np.float32),
            sh_coefficients=((colors - 0.5) / SH_C0)[:, :, None],
        )

    def update_gaussian(
        self,
        index: int,
        *,
        mean: np.ndarray | None = None,
        log_scale: np.ndarray | None = None,
        opacity_logit: float | None = None,
    ) -> None:
        if not 0 <= index < len(self):
            raise IndexError(f"Gaussian index {index} is out of range.")
        if mean is not None:
            self.means[index] = np.asarray(mean, dtype=np.float32)
        if log_scale is not None:
            self.log_scales[index] = np.asarray(log_scale, dtype=np.float32)
        if opacity_logit is not None:
            self.opacity_logits[index, 0] = float(opacity_logit)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            means=self.means,
            log_scales=self.log_scales,
            quaternions=self.quaternions,
            opacity_logits=self.opacity_logits,
            sh_coefficients=self.sh_coefficients,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "GaussianMap":
        with np.load(path) as archive:
            return cls(**{name: archive[name] for name in archive.files})

    def export_ply(self, path: str | Path) -> None:
        """Exports a readable subset of the Gaussian Splatting PLY convention."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        with path.open("w", encoding="ascii", newline="\n") as file:
            file.write("ply\nformat ascii 1.0\n")
            file.write(f"element vertex {len(self)}\n")
            for name in names:
                file.write(f"property float {name}\n")
            file.write("end_header\n")
            for index in range(len(self)):
                values = np.concatenate((
                    self.means[index],
                    self.sh_coefficients[index, :, 0],
                    self.opacity_logits[index],
                    self.log_scales[index],
                    self.quaternions[index],
                ))
                file.write(" ".join(f"{value:.8g}" for value in values) + "\n")
