"""Small, dependency-free readers for COLMAP sparse reconstruction files.

COLMAP stores camera poses as world-to-camera transforms: x_camera = R *
x_world + t.  The readers retain that convention and expose camera-to-world
matrices for rendering code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_PARAMETER_COUNTS = {name: parameter_count for name, parameter_count in CAMERA_MODELS.values()}


@dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    parameters: np.ndarray

    @property
    def focal_lengths(self) -> tuple[float, float]:
        if self.model.startswith("SIMPLE") or self.model in {"RADIAL", "FOV"}:
            return float(self.parameters[0]), float(self.parameters[0])
        return float(self.parameters[0]), float(self.parameters[1])

    @property
    def principal_point(self) -> tuple[float, float]:
        if self.model.startswith("SIMPLE") or self.model in {"RADIAL", "FOV"}:
            return float(self.parameters[1]), float(self.parameters[2])
        return float(self.parameters[2]), float(self.parameters[3])


@dataclass(frozen=True)
class Image:
    id: int
    camera_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray

    @property
    def world_to_camera(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = qvec_to_rotation_matrix(self.qvec)
        result[:3, 3] = self.tvec
        return result

    @property
    def camera_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera)


@dataclass(frozen=True)
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float


@dataclass(frozen=True)
class SparseModel:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: dict[int, Point3D]


def qvec_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    """Converts COLMAP's wxyz unit quaternion to a 3 by 3 rotation matrix."""
    qvec = np.asarray(qvec, dtype=np.float64).copy()
    if qvec.shape != (4,):
        raise ValueError("COLMAP quaternion must have shape (4,), in wxyz order.")
    qvec /= np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _read_exact(file: object, size: int) -> bytes:
    data = file.read(size)  # type: ignore[attr-defined]
    if len(data) != size:
        raise ValueError("Unexpected end of a COLMAP binary file.")
    return data


def _unpack(file: object, format_string: str) -> tuple[object, ...]:
    return struct.unpack("<" + format_string, _read_exact(file, struct.calcsize("<" + format_string)))


def _read_c_string(file: object) -> str:
    data = bytearray()
    while True:
        character = _read_exact(file, 1)
        if character == b"\0":
            return data.decode("utf-8")
        data.extend(character)


def read_cameras_binary(path: str | Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with Path(path).open("rb") as file:
        for _ in range(_unpack(file, "Q")[0]):
            camera_id, model_id, width, height = _unpack(file, "IIQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            model, parameter_count = CAMERA_MODELS[model_id]
            parameters = np.asarray(_unpack(file, "d" * parameter_count), dtype=np.float64)
            cameras[camera_id] = Camera(camera_id, model, width, height, parameters)
    return cameras


def read_images_binary(path: str | Path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with Path(path).open("rb") as file:
        for _ in range(_unpack(file, "Q")[0]):
            fields = _unpack(file, "IdddddddI")
            image_id, *pose, camera_id = fields
            images[image_id] = Image(image_id, camera_id, _read_c_string(file), np.asarray(pose[:4]), np.asarray(pose[4:]))
            point_count = _unpack(file, "Q")[0]
            _read_exact(file, point_count * struct.calcsize("<ddq"))
    return images


def read_points3d_binary(path: str | Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    with Path(path).open("rb") as file:
        for _ in range(_unpack(file, "Q")[0]):
            point_id, x, y, z, red, green, blue, error = _unpack(file, "QdddBBBd")
            track_length = _unpack(file, "Q")[0]
            _read_exact(file, track_length * struct.calcsize("<II"))
            points[point_id] = Point3D(point_id, np.array([x, y, z]), np.array([red, green, blue], dtype=np.uint8), error)
    return points


def _data_lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def read_cameras_text(path: str | Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _data_lines(Path(path)):
        fields = line.split()
        camera_id, model, width, height = fields[:4]
        parameters = np.asarray([float(value) for value in fields[4:]], dtype=np.float64)
        expected = CAMERA_MODEL_PARAMETER_COUNTS.get(model)
        if expected is None or len(parameters) != expected:
            raise ValueError(f"Invalid parameters for COLMAP camera model {model}.")
        cameras[int(camera_id)] = Camera(int(camera_id), model, int(width), int(height), parameters)
    return cameras


def read_images_text(path: str | Path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    lines = iter(_data_lines(Path(path)))
    for metadata in lines:
        fields = metadata.split()
        if len(fields) != 10:
            raise ValueError("Expected an image metadata line in images.txt.")
        image_id = int(fields[0])
        images[image_id] = Image(image_id, int(fields[8]), fields[9], np.asarray([float(value) for value in fields[1:5]]), np.asarray([float(value) for value in fields[5:8]]))
        next(lines, None)  # COLMAP's following line lists 2D observations, which are not needed here.
    return images


def read_points3d_text(path: str | Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    for line in _data_lines(Path(path)):
        fields = line.split()
        point_id = int(fields[0])
        points[point_id] = Point3D(point_id, np.asarray([float(value) for value in fields[1:4]]), np.asarray([int(value) for value in fields[4:7]], dtype=np.uint8), float(fields[7]))
    return points


def load_sparse_model(directory: str | Path) -> SparseModel:
    """Loads a COLMAP sparse model directory in binary or text form."""
    directory = Path(directory)
    binary_paths = [directory / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    text_paths = [directory / name for name in ("cameras.txt", "images.txt", "points3D.txt")]
    if all(path.is_file() for path in binary_paths):
        return SparseModel(read_cameras_binary(binary_paths[0]), read_images_binary(binary_paths[1]), read_points3d_binary(binary_paths[2]))
    if all(path.is_file() for path in text_paths):
        return SparseModel(read_cameras_text(text_paths[0]), read_images_text(text_paths[1]), read_points3d_text(text_paths[2]))
    raise FileNotFoundError(f"Expected cameras/images/points3D .bin or .txt files in {directory}.")
