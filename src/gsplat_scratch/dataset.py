"""Training views assembled from a COLMAP sparse model and its RGB images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PillowImage
import torch

from .colmap import SparseModel
from .renderer import RenderCamera


@dataclass(frozen=True)
class TrainingView:
    name: str
    camera: RenderCamera
    target: torch.Tensor


def load_training_views(model: SparseModel, image_directory: str | Path, device: torch.device | str, max_resolution: int | None = 160) -> list[TrainingView]:
    """Loads RGB targets and scales intrinsics when reference rendering downsizes."""
    image_directory = Path(image_directory)
    views: list[TrainingView] = []
    for image in model.images.values():
        source_path = image_directory / image.name
        if not source_path.is_file():
            raise FileNotFoundError(f"COLMAP image is missing: {source_path}")
        with PillowImage.open(source_path) as source:
            source = source.convert("RGB")
            original_width, original_height = source.size
            scale = 1.0 if max_resolution is None else min(1.0, max_resolution / max(original_width, original_height))
            width, height = round(original_width * scale), round(original_height * scale)
            if (width, height) != source.size:
                source = source.resize((width, height), PillowImage.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.asarray(source, dtype=np.float32) / 255.0).to(device)
        camera = RenderCamera.from_colmap(model.cameras[image.camera_id], image, device)
        if camera.width != original_width or camera.height != original_height:
            raise ValueError(f"COLMAP camera dimensions do not match {source_path}.")
        camera = RenderCamera(width, height, camera.fx * scale, camera.fy * scale, camera.cx * scale, camera.cy * scale, camera.world_to_camera)
        views.append(TrainingView(image.name, camera, pixels))
    if not views:
        raise ValueError("COLMAP model contains no registered images.")
    return views
