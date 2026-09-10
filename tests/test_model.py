import tempfile
import unittest
from pathlib import Path

import numpy as np

from gsplat_scratch.model import GaussianMap


class GaussianMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.map = GaussianMap.from_points(
            np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )

    def test_parameterisation_is_valid(self) -> None:
        np.testing.assert_allclose(self.map.scales, 0.03, rtol=1e-5)
        self.assertTrue(np.all((self.map.opacity > 0) & (self.map.opacity < 1)))
        np.testing.assert_allclose(self.map.base_colors, [[1, 0, 0], [0, 1, 0]], atol=1e-6)

    def test_npz_round_trip_and_ply_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            npz_path = Path(directory) / "map.npz"
            ply_path = Path(directory) / "map.ply"
            self.map.save_npz(npz_path)
            restored = GaussianMap.load_npz(npz_path)
            restored.export_ply(ply_path)
            np.testing.assert_array_equal(restored.means, self.map.means)
            self.assertIn("element vertex 2", ply_path.read_text(encoding="ascii"))

    def test_update_gaussian(self) -> None:
        self.map.update_gaussian(1, mean=np.array([4, 5, 6]), log_scale=np.array([-2, -3, -4]), opacity_logit=-1)
        np.testing.assert_array_equal(self.map.means[1], [4, 5, 6])
        np.testing.assert_array_equal(self.map.log_scales[1], [-2, -3, -4])
        self.assertAlmostEqual(float(self.map.opacity_logits[1, 0]), -1)


if __name__ == "__main__":
    unittest.main()
