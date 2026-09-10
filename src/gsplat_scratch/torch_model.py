"""Torch-owned Gaussian parameters used by the optimiser and renderer."""

from __future__ import annotations

import torch
from torch import nn

from .model import GaussianMap


class TorchGaussianParameters(nn.Module):
    """Differentiable counterpart of :class:`GaussianMap`.

    The unconstrained storage matches the paper: log-scales, opacity logits,
    and unnormalised quaternions are optimised directly.
    """

    def __init__(self, gaussian_map: GaussianMap, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.means = nn.Parameter(torch.as_tensor(gaussian_map.means, dtype=torch.float32, device=device))
        self.log_scales = nn.Parameter(torch.as_tensor(gaussian_map.log_scales, dtype=torch.float32, device=device))
        self.quaternions = nn.Parameter(torch.as_tensor(gaussian_map.quaternions, dtype=torch.float32, device=device))
        self.opacity_logits = nn.Parameter(torch.as_tensor(gaussian_map.opacity_logits, dtype=torch.float32, device=device))
        self.sh_coefficients = nn.Parameter(torch.as_tensor(gaussian_map.sh_coefficients, dtype=torch.float32, device=device))

    def to_gaussian_map(self) -> GaussianMap:
        return GaussianMap(
            means=self.means.detach().cpu().numpy(),
            log_scales=self.log_scales.detach().cpu().numpy(),
            quaternions=self.quaternions.detach().cpu().numpy(),
            opacity_logits=self.opacity_logits.detach().cpu().numpy(),
            sh_coefficients=self.sh_coefficients.detach().cpu().numpy(),
        )
