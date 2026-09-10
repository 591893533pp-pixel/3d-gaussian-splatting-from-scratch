"""Image quality evaluation for rendered novel views."""

from __future__ import annotations

import math

import torch

from .dataset import TrainingView
from .renderer import render_reference, render_tiled
from .torch_model import TorchGaussianParameters
from .trainer import dssim


def psnr(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (image - target).square().mean().clamp_min(1e-10)
    return -10 * torch.log10(mse)


@torch.no_grad()
def evaluate_views(parameters: TorchGaussianParameters, views: list[TrainingView], renderer: str = "tiled") -> dict[str, float]:
    render = render_tiled if renderer == "tiled" else render_reference
    values = []
    for view in views:
        image = render(parameters.means, parameters.log_scales, parameters.quaternions, parameters.opacity_logits, parameters.sh_coefficients, view.camera)
        values.append((float(psnr(image, view.target)), float(1 - dssim(image, view.target) * 2)))
    return {"psnr": sum(value[0] for value in values) / len(values), "ssim": sum(value[1] for value in values) / len(values), "views": len(values)}
