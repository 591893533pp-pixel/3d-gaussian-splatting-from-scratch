"""Reference differentiable 3D Gaussian rasteriser written in PyTorch.

This implementation intentionally evaluates every Gaussian over every pixel.
It is compact enough to audit and produces gradients for optimisation.  The
later CUDA tile rasteriser will preserve these equations while avoiding this
O(NHW) reference cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from .colmap import Camera, Image
from .model import SH_C0


@dataclass(frozen=True)
class RenderCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: torch.Tensor

    @classmethod
    def from_colmap(cls, camera: Camera, image: Image, device: torch.device | str = "cpu") -> "RenderCamera":
        fx, fy = camera.focal_lengths
        cx, cy = camera.principal_point
        return cls(camera.width, camera.height, fx, fy, cx, cy, torch.as_tensor(image.world_to_camera, dtype=torch.float32, device=device))

    @classmethod
    def identity(cls, width: int, height: int, focal_length: float, device: torch.device | str = "cpu") -> "RenderCamera":
        return cls(width, height, focal_length, focal_length, (width - 1) / 2, (height - 1) / 2, torch.eye(4, dtype=torch.float32, device=device))

    def to(self, device: torch.device | str) -> "RenderCamera":
        return RenderCamera(self.width, self.height, self.fx, self.fy, self.cx, self.cy, self.world_to_camera.to(device))


@dataclass(frozen=True)
class ProjectedGaussians:
    means_2d: torch.Tensor
    covariances_2d: torch.Tensor
    depths: torch.Tensor
    radii: torch.Tensor
    visible: torch.Tensor


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Converts N wxyz quaternions into N rotation matrices."""
    quaternions = functional.normalize(quaternions, dim=-1, eps=1e-8)
    w, x, y, z = quaternions.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y.square() + z.square()), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x.square() + z.square()), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x.square() + y.square()),
    ), dim=-1).reshape(-1, 3, 3)


def project_gaussians(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quaternions: torch.Tensor,
    camera: RenderCamera,
    minimum_variance: float = 0.3,
    near_plane: float = 0.01,
) -> ProjectedGaussians:
    """Projects 3D Gaussian covariance with the perspective Jacobian."""
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape (N, 3).")
    if log_scales.shape != means.shape or quaternions.shape != (len(means), 4):
        raise ValueError("Gaussian parameter shapes do not agree.")
    camera = camera.to(means.device)
    rotation_cw = camera.world_to_camera[:3, :3]
    translation_cw = camera.world_to_camera[:3, 3]
    means_camera = means @ rotation_cw.T + translation_cw
    x, y, z = means_camera.unbind(dim=-1)
    safe_z = z.clamp_min(near_plane)
    means_2d = torch.stack((camera.fx * x / safe_z + camera.cx, camera.fy * y / safe_z + camera.cy), dim=-1)

    scales_squared = torch.exp(log_scales * 2)
    rotation_gs = quaternion_to_rotation_matrix(quaternions)
    covariance_world = (rotation_gs * scales_squared.unsqueeze(-2)) @ rotation_gs.transpose(-1, -2)
    covariance_camera = rotation_cw.unsqueeze(0) @ covariance_world @ rotation_cw.T.unsqueeze(0)
    zeros = torch.zeros_like(z)
    jacobian = torch.stack((
        torch.stack((torch.full_like(z, camera.fx) / safe_z, zeros, -camera.fx * x / safe_z.square()), dim=-1),
        torch.stack((zeros, torch.full_like(z, camera.fy) / safe_z, -camera.fy * y / safe_z.square()), dim=-1),
    ), dim=-2)
    covariance_2d = jacobian @ covariance_camera @ jacobian.transpose(-1, -2)
    covariance_2d = covariance_2d + torch.eye(2, dtype=means.dtype, device=means.device).unsqueeze(0) * minimum_variance
    radii = 3 * torch.sqrt(torch.linalg.eigvalsh(covariance_2d).amax(dim=-1).clamp_min(0))
    visible = (z > near_plane) & (means_2d[:, 0] + radii >= 0) & (means_2d[:, 0] - radii < camera.width) & (means_2d[:, 1] + radii >= 0) & (means_2d[:, 1] - radii < camera.height)
    return ProjectedGaussians(means_2d, covariance_2d, z, radii, visible)


def sh_degree_zero_colours(sh_coefficients: torch.Tensor) -> torch.Tensor:
    if sh_coefficients.ndim != 3 or sh_coefficients.shape[1] != 3:
        raise ValueError("sh_coefficients must have shape (N, 3, K).")
    return (0.5 + SH_C0 * sh_coefficients[:, :, 0]).clamp(0.0, 1.0)


def render_reference(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quaternions: torch.Tensor,
    opacity_logits: torch.Tensor,
    sh_coefficients: torch.Tensor,
    camera: RenderCamera,
    background: torch.Tensor | None = None,
    return_projection: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ProjectedGaussians]:
    """Renders a view with differentiable front-to-back alpha compositing."""
    projected = project_gaussians(means, log_scales, quaternions, camera)
    height, width = camera.height, camera.width
    device, dtype = means.device, means.dtype
    if background is None:
        background = torch.zeros(3, dtype=dtype, device=device)
    background = background.to(device=device, dtype=dtype).reshape(3)
    ys, xs = torch.meshgrid(torch.arange(height, dtype=dtype, device=device), torch.arange(width, dtype=dtype, device=device), indexing="ij")
    pixel_coordinates = torch.stack((xs, ys), dim=-1)
    delta = pixel_coordinates.unsqueeze(0) - projected.means_2d[:, None, None, :]
    inverse_covariance = torch.linalg.inv(projected.covariances_2d)
    mahalanobis = torch.einsum("nhwi,nij,nhwj->nhw", delta, inverse_covariance, delta)
    alpha = torch.sigmoid(opacity_logits.reshape(-1))[:, None, None] * torch.exp(-0.5 * mahalanobis)
    within_radius = mahalanobis <= 9.0
    alpha = torch.where(projected.visible[:, None, None] & within_radius, alpha, torch.zeros_like(alpha)).clamp_max(0.999)
    order = torch.argsort(projected.depths)
    colours = sh_degree_zero_colours(sh_coefficients)[order]
    transmittance = torch.ones((height, width), dtype=dtype, device=device)
    image = torch.zeros((height, width, 3), dtype=dtype, device=device)
    for gaussian_alpha, colour in zip(alpha[order], colours):
        contribution = transmittance * gaussian_alpha
        image = image + contribution.unsqueeze(-1) * colour
        transmittance = transmittance * (1 - gaussian_alpha)
    image = image + transmittance.unsqueeze(-1) * background
    return (image, projected) if return_projection else image
