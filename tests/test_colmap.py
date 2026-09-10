import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gsplat_scratch.colmap import load_sparse_model, qvec_to_rotation_matrix
from gsplat_scratch.initialization import initialise_from_sparse_model, nearest_neighbor_distances


class ColmapTests(unittest.TestCase):
    def _write_text_model(self, directory: Path) -> None:
        (directory / "cameras.txt").write_text("# Camera list\n1 PINHOLE 640 480 500 510 320 240\n", encoding="utf-8")
        (directory / "images.txt").write_text("# Image list\n1 1 0 0 0 0 0 0 1 frame.png\n0 0 -1\n", encoding="utf-8")
        (directory / "points3D.txt").write_text("# Point list\n1 0 0 0 255 0 0 0.1 1 0\n2 1 0 0 0 255 0 0.2 1 1\n", encoding="utf-8")

    def test_load_text_model_and_initialise_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._write_text_model(Path(directory))
            model = load_sparse_model(directory)
            self.assertEqual(model.cameras[1].focal_lengths, (500.0, 510.0))
            self.assertEqual(model.cameras[1].principal_point, (320.0, 240.0))
            np.testing.assert_allclose(model.images[1].camera_to_world, np.eye(4))
            gaussian_map = initialise_from_sparse_model(model)
            self.assertEqual(len(gaussian_map), 2)
            np.testing.assert_allclose(gaussian_map.scales, 1.0)
            np.testing.assert_allclose(gaussian_map.base_colors, [[1, 0, 0], [0, 1, 0]], atol=1e-6)

    def test_read_binary_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with (directory / "cameras.bin").open("wb") as file:
                file.write(struct.pack("<QIIQQdddd", 1, 5, 1, 640, 480, 500, 510, 320, 240))
            with (directory / "images.bin").open("wb") as file:
                file.write(struct.pack("<QIdddddddI", 1, 3, 1, 0, 0, 0, 1, 2, 3, 5))
                file.write(b"frame.png\0")
                file.write(struct.pack("<Qddq", 1, 30.0, 40.0, 9))
            with (directory / "points3D.bin").open("wb") as file:
                file.write(struct.pack("<QQdddBBBdQII", 1, 9, 1, 2, 3, 4, 5, 6, 0.2, 1, 3, 0))
            model = load_sparse_model(directory)
            self.assertEqual(model.images[3].name, "frame.png")
            np.testing.assert_allclose(model.images[3].world_to_camera[:3, 3], [1, 2, 3])
            np.testing.assert_array_equal(model.points[9].rgb, [4, 5, 6])

    def test_quaternion_and_nearest_neighbour(self) -> None:
        rotation = qvec_to_rotation_matrix(np.array([0.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(rotation, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], atol=1e-7)
        np.testing.assert_allclose(nearest_neighbor_distances(np.array([[0, 0, 0], [3, 0, 0], [10, 0, 0]])), [3, 3, 7])


if __name__ == "__main__":
    unittest.main()
