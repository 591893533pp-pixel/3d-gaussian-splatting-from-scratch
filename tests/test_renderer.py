import unittest

import numpy as np
import torch

from gsplat_scratch.model import GaussianMap, SH_C0
from gsplat_scratch.renderer import RenderCamera, project_gaussians, render_reference
from gsplat_scratch.torch_model import TorchGaussianParameters


def _sh_colour(red: float, green: float, blue: float) -> torch.Tensor:
    colour = torch.tensor([red, green, blue], dtype=torch.float32)
    return ((colour - 0.5) / SH_C0).reshape(1, 3, 1)


class ReferenceRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = RenderCamera.identity(width=33, height=33, focal_length=32)

    def test_central_point_projects_to_principal_point(self) -> None:
        projected = project_gaussians(torch.tensor([[0.0, 0.0, 2.0]]), torch.zeros((1, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), self.camera)
        torch.testing.assert_close(projected.means_2d, torch.tensor([[16.0, 16.0]]))
        self.assertTrue(bool(projected.visible[0]))

    def test_nearer_gaussian_occludes_farther_one(self) -> None:
        image = render_reference(
            means=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]),
            log_scales=torch.full((2, 3), -1.5),
            quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            opacity_logits=torch.tensor([[8.0], [8.0]]),
            sh_coefficients=torch.cat((_sh_colour(1, 0, 0), _sh_colour(0, 1, 0))),
            camera=self.camera,
        )
        centre = image[16, 16]
        self.assertGreater(float(centre[0]), 0.95)
        self.assertLess(float(centre[1]), 0.05)

    def test_renderer_backpropagates_to_parameters(self) -> None:
        map_ = GaussianMap.from_points(np.array([[0.0, 0.0, 2.5], [0.2, 0.1, 3.0]], dtype=np.float32), np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), scale=0.2)
        parameters = TorchGaussianParameters(map_)
        image = render_reference(parameters.means, parameters.log_scales, parameters.quaternions, parameters.opacity_logits, parameters.sh_coefficients, self.camera)
        image[..., 0].mean().backward()
        self.assertTrue(torch.isfinite(parameters.means.grad).all())
        self.assertGreater(float(parameters.means.grad.abs().sum()), 0.0)
        self.assertGreater(float(parameters.opacity_logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
