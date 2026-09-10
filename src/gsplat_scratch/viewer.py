"""A small, inspectable editor for a Gaussian map."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider

from .model import GaussianMap


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Builds a rotation matrix from a normalised wxyz quaternion."""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


class GaussianMapEditor:
    """Matplotlib editor for inspecting and adjusting individual Gaussians."""

    def __init__(self, gaussian_map: GaussianMap, output_path: str | None = None, max_visible: int = 96) -> None:
        self.map = gaussian_map
        self.output_path = output_path
        self.max_visible = max_visible
        self.selected = 0
        self.figure = plt.figure(figsize=(12, 10))
        self.axes = self.figure.add_axes((0.05, 0.48, 0.9, 0.46), projection="3d")
        self._make_controls()
        self._draw()

    def _make_controls(self) -> None:
        bounds = np.ptp(self.map.means, axis=0)
        self.position_limit = max(float(np.max(bounds)) * 0.75, 0.5)
        self.sliders: dict[str, Slider] = {}
        specs = [
            ("index", "Gaussian", 0, max(len(self.map) - 1, 1), 0, 0.39),
            ("x", "X", -self.position_limit, self.position_limit, 0, 0.34),
            ("y", "Y", -self.position_limit, self.position_limit, 0, 0.29),
            ("z", "Z", -self.position_limit, self.position_limit, 0, 0.24),
            ("sx", "log scale X", -6, 1, -3, 0.19),
            ("sy", "log scale Y", -6, 1, -3, 0.14),
            ("sz", "log scale Z", -6, 1, -3, 0.09),
            ("opacity", "opacity logit", -8, 8, 2, 0.04),
        ]
        for key, label, low, high, initial, y in specs:
            axis = self.figure.add_axes((0.16, y, 0.57, 0.022))
            slider = Slider(axis, label, low, high, valinit=initial, valstep=1 if key == "index" else None)
            slider.on_changed(self._on_slider)
            self.sliders[key] = slider
        save_axis = self.figure.add_axes((0.78, 0.36, 0.14, 0.05))
        self.save_button = Button(save_axis, "Save map")
        self.save_button.on_clicked(self._save)

    def _set_slider_values(self) -> None:
        gaussian = self.map
        index = self.selected
        values = {
            "x": gaussian.means[index, 0], "y": gaussian.means[index, 1], "z": gaussian.means[index, 2],
            "sx": gaussian.log_scales[index, 0], "sy": gaussian.log_scales[index, 1], "sz": gaussian.log_scales[index, 2],
            "opacity": gaussian.opacity_logits[index, 0],
        }
        for key, value in values.items():
            self.sliders[key].eventson = False
            self.sliders[key].set_val(float(value))
            self.sliders[key].eventson = True

    def _on_slider(self, _value: float) -> None:
        new_index = int(self.sliders["index"].val)
        if new_index != self.selected:
            self.selected = new_index
            self._set_slider_values()
        else:
            self.map.update_gaussian(
                self.selected,
                mean=np.array([self.sliders[key].val for key in ("x", "y", "z")]),
                log_scale=np.array([self.sliders[key].val for key in ("sx", "sy", "sz")]),
                opacity_logit=self.sliders["opacity"].val,
            )
        self._draw()

    def _draw_ellipsoid(self, index: int, resolution: int = 12) -> None:
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        sphere = np.stack((np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), np.outer(np.ones_like(u), np.cos(v))))
        transformed = quaternion_to_matrix(self.map.quaternions[index]) @ (sphere.reshape(3, -1) * self.map.scales[index, :, None])
        transformed += self.map.means[index, :, None]
        x, y, z = transformed.reshape(3, resolution, resolution)
        color = self.map.base_colors[index]
        alpha = float(np.clip(self.map.opacity[index, 0], 0.08, 0.92))
        if index == self.selected:
            color = np.array([1.0, 0.75, 0.05])
            alpha = 0.95
        self.axes.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.15, edgecolor="white" if index == self.selected else "none")

    def _draw(self) -> None:
        self.axes.clear()
        visible = min(len(self.map), self.max_visible)
        centers = self.map.means[:visible]
        self.axes.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=5, c=self.map.base_colors[:visible], alpha=0.45)
        for index in range(visible):
            self._draw_ellipsoid(index)
        center = np.mean(centers, axis=0)
        radius = max(float(np.max(np.ptp(centers, axis=0))) * 0.6, 0.7)
        self.axes.set(xlim=(center[0] - radius, center[0] + radius), ylim=(center[1] - radius, center[1] + radius), zlim=(center[2] - radius, center[2] + radius), xlabel="X", ylabel="Y", zlabel="Z")
        self.axes.set_box_aspect((1, 1, 1))
        self.axes.set_title(f"Gaussian map | selected #{self.selected} | {len(self.map)} Gaussians")
        self.figure.canvas.draw_idle()

    def _save(self, _event: object) -> None:
        if self.output_path is None:
            print("No --output path supplied; map was not saved.")
            return
        self.map.save_npz(self.output_path)
        print(f"Saved edited map to {self.output_path}")

    def show(self) -> None:
        plt.show()
