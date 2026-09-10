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

SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


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


def sh_colours(sh_coefficients: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Evaluates real spherical harmonics through degree three."""
    if sh_coefficients.ndim != 3 or sh_coefficients.shape[1] != 3:
        raise ValueError("sh_coefficients must have shape (N, 3, K).")
    if sh_coefficients.shape[2] not in {1, 4, 9, 16}:
        raise ValueError("SH coefficient count must be 1, 4, 9, or 16.")
    x, y, z = directions.unbind(dim=-1)
    result = SH_C0 * sh_coefficients[:, :, 0]
    if sh_coefficients.shape[2] >= 4:
        result = result - SH_C1 * y[:, None] * sh_coefficients[:, :, 1] + SH_C1 * z[:, None] * sh_coefficients[:, :, 2] - SH_C1 * x[:, None] * sh_coefficients[:, :, 3]
    if sh_coefficients.shape[2] >= 9:
        xx, yy, zz = x.square(), y.square(), z.square()
        result = result + SH_C2[0] * (x * y)[:, None] * sh_coefficients[:, :, 4] + SH_C2[1] * (y * z)[:, None] * sh_coefficients[:, :, 5] + SH_C2[2] * (2 * zz - xx - yy)[:, None] * sh_coefficients[:, :, 6] + SH_C2[3] * (x * z)[:, None] * sh_coefficients[:, :, 7] + SH_C2[4] * (xx - yy)[:, None] * sh_coefficients[:, :, 8]
    if sh_coefficients.shape[2] >= 16:
        xx, yy, zz = x.square(), y.square(), z.square()
        result = result + SH_C3[0] * (y * (3 * xx - yy))[:, None] * sh_coefficients[:, :, 9] + SH_C3[1] * (x * y * z)[:, None] * sh_coefficients[:, :, 10] + SH_C3[2] * (y * (4 * zz - xx - yy))[:, None] * sh_coefficients[:, :, 11] + SH_C3[3] * (z * (2 * zz - 3 * xx - 3 * yy))[:, None] * sh_coefficients[:, :, 12] + SH_C3[4] * (x * (4 * zz - xx - yy))[:, None] * sh_coefficients[:, :, 13] + SH_C3[5] * (z * (xx - yy))[:, None] * sh_coefficients[:, :, 14] + SH_C3[6] * (x * (xx - 3 * yy))[:, None] * sh_coefficients[:, :, 15]
    return (0.5 + result).clamp(0.0, 1.0)


def _view_directions(means: torch.Tensor, camera: RenderCamera) -> torch.Tensor:
    camera_center = torch.linalg.inv(camera.to(means.device).world_to_camera)[:3, 3]
    return functional.normalize(camera_center.unsqueeze(0) - means, dim=-1, eps=1e-8)


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
    colours = sh_colours(sh_coefficients, _view_directions(means, camera))[order]
    transmittance = torch.ones((height, width), dtype=dtype, device=device)
    image = torch.zeros((height, width, 3), dtype=dtype, device=device)
    for gaussian_alpha, colour in zip(alpha[order], colours):
        contribution = transmittance * gaussian_alpha
        image = image + contribution.unsqueeze(-1) * colour
        transmittance = transmittance * (1 - gaussian_alpha)
    image = image + transmittance.unsqueeze(-1) * background
    return (image, projected) if return_projection else image


def render_tiled(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quaternions: torch.Tensor,
    opacity_logits: torch.Tensor,
    sh_coefficients: torch.Tensor,
    camera: RenderCamera,
    background: torch.Tensor | None = None,
    tile_size: int = 16,
    return_projection: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ProjectedGaussians]:
    """CUDA-friendly tile culling variant of :func:`render_reference`.

    The math is identical to the reference renderer, but each tile evaluates
    only Gaussians whose 3-sigma bounds overlap that tile. PyTorch dispatches
    every tensor operation on CUDA when the inputs are CUDA tensors.
    """
    projected = project_gaussians(means, log_scales, quaternions, camera)
    height, width = camera.height, camera.width
    device, dtype = means.device, means.dtype
    if background is None:
        background = torch.zeros(3, dtype=dtype, device=device)
    background = background.to(device=device, dtype=dtype).reshape(3)
    inverse_covariance = torch.linalg.inv(projected.covariances_2d)
    colours = sh_colours(sh_coefficients, _view_directions(means, camera))
    rows: list[torch.Tensor] = []
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        tiles: list[torch.Tensor] = []
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            overlaps = projected.visible & (projected.means_2d[:, 0] + projected.radii >= left) & (projected.means_2d[:, 0] - projected.radii < right) & (projected.means_2d[:, 1] + projected.radii >= top) & (projected.means_2d[:, 1] - projected.radii < bottom)
            indices = torch.nonzero(overlaps, as_tuple=False).squeeze(-1)
            tile_height, tile_width = bottom - top, right - left
            if len(indices) == 0:
                tiles.append(background.expand(tile_height, tile_width, 3))
                continue
            ys, xs = torch.meshgrid(torch.arange(top, bottom, dtype=dtype, device=device), torch.arange(left, right, dtype=dtype, device=device), indexing="ij")
            coordinates = torch.stack((xs, ys), dim=-1)
            delta = coordinates.unsqueeze(0) - projected.means_2d[indices, None, None, :]
            mahalanobis = torch.einsum("nhwi,nij,nhwj->nhw", delta, inverse_covariance[indices], delta)
            alpha = torch.sigmoid(opacity_logits.reshape(-1)[indices])[:, None, None] * torch.exp(-0.5 * mahalanobis)
            alpha = torch.where(mahalanobis <= 9.0, alpha, torch.zeros_like(alpha)).clamp_max(0.999)
            order = torch.argsort(projected.depths[indices])
            transmittance = torch.ones((tile_height, tile_width), dtype=dtype, device=device)
            tile_image = torch.zeros((tile_height, tile_width, 3), dtype=dtype, device=device)
            for gaussian_alpha, colour in zip(alpha[order], colours[indices][order]):
                contribution = transmittance * gaussian_alpha
                tile_image = tile_image + contribution.unsqueeze(-1) * colour
                transmittance = transmittance * (1 - gaussian_alpha)
            tiles.append(tile_image + transmittance.unsqueeze(-1) * background)
        rows.append(torch.cat(tiles, dim=1))
    image = torch.cat(rows, dim=0)
    return (image, projected) if return_projection else image
