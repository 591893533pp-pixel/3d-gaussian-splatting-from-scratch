import unittest

import numpy as np
import torch

from gsplat_scratch.dataset import TrainingView
from gsplat_scratch.density import DensityController
from gsplat_scratch.model import GaussianMap
from gsplat_scratch.renderer import RenderCamera, render_reference
from gsplat_scratch.trainer import GaussianTrainer, TrainingConfig, dssim
from gsplat_scratch.torch_model import TorchGaussianParameters


class TrainerTests(unittest.TestCase):
    def test_dssim_is_zero_for_identical_images(self) -> None:
        image = torch.rand(8, 8, 3)
        self.assertLess(float(dssim(image, image)), 1e-6)

    def test_density_controller_clones_and_prunes(self) -> None:
        gaussian_map = GaussianMap.from_points(np.array([[0, 0, 1], [1, 0, 1]], dtype=np.float32), np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), scale=0.01)
        gaussian_map.opacity_logits[1] = -20
        controller = DensityController(gradient_threshold=0.1, small_scale=0.1)
        controller.accumulate(np.array([[1, 0], [0, 0]], dtype=np.float32), np.array([True, True]))
        refined = controller.refine(gaussian_map)
        self.assertEqual(len(refined), 2)

    def test_density_controller_replaces_large_gaussian_when_splitting(self) -> None:
        gaussian_map = GaussianMap.from_points(np.array([[0, 0, 1]], dtype=np.float32), np.array([[1, 0, 0]], dtype=np.float32), scale=0.5)
        controller = DensityController(gradient_threshold=0.1, small_scale=0.1)
        controller.accumulate(np.array([[1, 0]], dtype=np.float32), np.array([True]))
        refined = controller.refine(gaussian_map)
        self.assertEqual(len(refined), 2)
        self.assertTrue(np.all(refined.scales < gaussian_map.scales[0]))

    def test_training_step_reduces_simple_colour_loss(self) -> None:
        camera = RenderCamera.identity(17, 17, 16)
        target_map = GaussianMap.from_points(np.array([[0, 0, 2]], dtype=np.float32), np.array([[1, 0, 0]], dtype=np.float32), scale=0.2)
        with torch.no_grad():
            target_parameters = TorchGaussianParameters(target_map)
            target = render_reference(target_parameters.means, target_parameters.log_scales, target_parameters.quaternions, target_parameters.opacity_logits, target_parameters.sh_coefficients, camera)
        initial_map = GaussianMap.from_points(np.array([[0, 0, 2]], dtype=np.float32), np.array([[0, 1, 0]], dtype=np.float32), scale=0.2)
        trainer = GaussianTrainer(initial_map, [TrainingView("synthetic", camera, target)], TrainingConfig(iterations=12, dssim_weight=0, feature_lr=0.2, densify_from=100), "cpu")
        initial_loss = trainer.step(1)
        for iteration in range(2, 13):
            final_loss = trainer.step(iteration)
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
