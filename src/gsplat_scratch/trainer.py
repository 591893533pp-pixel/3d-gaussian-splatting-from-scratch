"""Optimisation loop for the reference 3DGS renderer."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch.nn import functional as functional

from .dataset import TrainingView
from .density import DensityController
from .model import GaussianMap
from .renderer import render_reference, render_tiled
from .torch_model import TorchGaussianParameters


def dssim(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable SSIM dissimilarity for HWC RGB tensors."""
    image = image.permute(2, 0, 1).unsqueeze(0)
    target = target.permute(2, 0, 1).unsqueeze(0)
    channels = image.shape[1]
    kernel = torch.ones((channels, 1, 3, 3), dtype=image.dtype, device=image.device) / 9.0
    mu_image = functional.conv2d(image, kernel, padding=1, groups=channels)
    mu_target = functional.conv2d(target, kernel, padding=1, groups=channels)
    sigma_image = functional.conv2d(image * image, kernel, padding=1, groups=channels) - mu_image.square()
    sigma_target = functional.conv2d(target * target, kernel, padding=1, groups=channels) - mu_target.square()
    covariance = functional.conv2d(image * target, kernel, padding=1, groups=channels) - mu_image * mu_target
    score = ((2 * mu_image * mu_target + 0.01) * (2 * covariance + 0.09)) / ((mu_image.square() + mu_target.square() + 0.01) * (sigma_image + sigma_target + 0.09))
    return (1 - score.mean()) / 2


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 1_000
    dssim_weight: float = 0.2
    position_lr: float = 1.6e-4
    feature_lr: float = 2.5e-3
    opacity_lr: float = 5e-2
    scale_lr: float = 5e-3
    rotation_lr: float = 1e-3
    densify_from: int = 300
    densify_until: int = 3_000
    densify_interval: int = 100
    renderer: str = "tiled"


class GaussianTrainer:
    def __init__(self, initial_map: GaussianMap, views: list[TrainingView], config: TrainingConfig, device: torch.device | str = "cuda") -> None:
        self.views = views
        self.config = config
        self.device = torch.device(device)
        self.parameters = TorchGaussianParameters(initial_map, self.device)
        self.density = DensityController()
        self.optimizer = self._make_optimizer()
        self.loss_history: list[float] = []

    def _make_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam([
            {"params": [self.parameters.means], "lr": self.config.position_lr, "name": "position"},
            {"params": [self.parameters.sh_coefficients], "lr": self.config.feature_lr, "name": "features"},
            {"params": [self.parameters.opacity_logits], "lr": self.config.opacity_lr, "name": "opacity"},
            {"params": [self.parameters.log_scales], "lr": self.config.scale_lr, "name": "scale"},
            {"params": [self.parameters.quaternions], "lr": self.config.rotation_lr, "name": "rotation"},
        ], eps=1e-15)

    def step(self, iteration: int) -> float:
        view = random.choice(self.views)
        self.optimizer.zero_grad(set_to_none=True)
        renderer = render_tiled if self.config.renderer == "tiled" else render_reference
        rendered, projected = renderer(self.parameters.means, self.parameters.log_scales, self.parameters.quaternions, self.parameters.opacity_logits, self.parameters.sh_coefficients, view.camera, return_projection=True)
        projected.means_2d.retain_grad()
        l1 = (rendered - view.target).abs().mean()
        loss = (1 - self.config.dssim_weight) * l1 + self.config.dssim_weight * dssim(rendered, view.target)
        loss.backward()
        if projected.means_2d.grad is not None:
            self.density.accumulate(projected.means_2d.grad.detach().cpu().numpy(), projected.visible.detach().cpu().numpy())
        self.optimizer.step()
        if self.config.densify_from <= iteration <= self.config.densify_until and iteration % self.config.densify_interval == 0:
            refined = self.density.refine(self.parameters.to_gaussian_map())
            self.parameters = TorchGaussianParameters(refined, self.device)
            self.optimizer = self._make_optimizer()
        value = float(loss.detach())
        self.loss_history.append(value)
        return value

    def train(self) -> GaussianMap:
        for iteration in range(1, self.config.iterations + 1):
            self.step(iteration)
        return self.parameters.to_gaussian_map()
