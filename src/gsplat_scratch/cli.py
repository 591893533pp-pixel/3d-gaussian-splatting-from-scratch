"""Command line entry points for the first 3DGS project milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from .model import GaussianMap
from .colmap import load_sparse_model
from .initialization import initialise_from_sparse_model
from .renderer import RenderCamera, render_reference
from .torch_model import TorchGaussianParameters
from .dataset import load_training_views
from .trainer import GaussianTrainer, TrainingConfig
from .metrics import evaluate_views
from .viewer import GaussianMapEditor


def make_demo_map(count: int, seed: int) -> GaussianMap:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radius = rng.uniform(0.3, 1.0, size=(count, 1))
    colors = np.clip(0.5 + 0.45 * directions, 0.0, 1.0)
    result = GaussianMap.from_points(directions * radius, colors, scale=0.06)
    result.log_scales += rng.normal(0.0, 0.35, size=(count, 3)).astype(np.float32)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="From-scratch 3D Gaussian Splatting utilities")
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo", help="Create an editable demonstration Gaussian map")
    demo.add_argument("output", type=Path)
    demo.add_argument("--count", type=int, default=48)
    demo.add_argument("--seed", type=int, default=7)
    view = subcommands.add_parser("view", help="Open a Gaussian-map editor")
    view.add_argument("input", type=Path)
    view.add_argument("--output", type=Path)
    view.add_argument("--max-visible", type=int, default=96)
    export = subcommands.add_parser("export-ply", help="Export a saved Gaussian map as ASCII PLY")
    export.add_argument("input", type=Path)
    export.add_argument("output", type=Path)
    import_colmap = subcommands.add_parser("import-colmap", help="Initialise a Gaussian map from COLMAP sparse reconstruction")
    import_colmap.add_argument("sparse_model", type=Path, help="Directory containing COLMAP cameras/images/points3D files")
    import_colmap.add_argument("output", type=Path)
    render_demo = subcommands.add_parser("render-demo", help="Render a small differentiable Gaussian scene to a PNG")
    render_demo.add_argument("output", type=Path)
    render_demo.add_argument("--size", type=int, default=128)
    train = subcommands.add_parser("train-colmap", help="Train the reference renderer on COLMAP images")
    train.add_argument("sparse_model", type=Path)
    train.add_argument("image_directory", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--iterations", type=int, default=500)
    train.add_argument("--max-resolution", type=int, default=128)
    train.add_argument("--max-gaussians", type=int, default=256)
    train.add_argument("--sh-degree", type=int, choices=range(4), default=3)
    train.add_argument("--renderer", choices=("reference", "tiled"), default="tiled")
    evaluate = subcommands.add_parser("evaluate-colmap", help="Evaluate a saved Gaussian map against COLMAP images")
    evaluate.add_argument("gaussian_map", type=Path)
    evaluate.add_argument("sparse_model", type=Path)
    evaluate.add_argument("image_directory", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--max-resolution", type=int, default=128)
    evaluate.add_argument("--renderer", choices=("reference", "tiled"), default="tiled")
    args = parser.parse_args()
    if args.command == "demo":
        make_demo_map(args.count, args.seed).save_npz(args.output)
        print(f"Created {args.count} Gaussians at {args.output}")
    elif args.command == "view":
        GaussianMapEditor(GaussianMap.load_npz(args.input), str(args.output) if args.output else None, args.max_visible).show()
    elif args.command == "import-colmap":
        model = load_sparse_model(args.sparse_model)
        initialise_from_sparse_model(model).save_npz(args.output)
        print(f"Imported {len(model.cameras)} cameras, {len(model.images)} images, and {len(model.points)} points to {args.output}")
    elif args.command == "render-demo":
        gaussian_map = make_demo_map(24, 7)
        gaussian_map.means[:, 2] += 3.0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        parameters = TorchGaussianParameters(gaussian_map, device=device)
        camera = RenderCamera.identity(args.size, args.size, focal_length=args.size, device=device)
        image = render_reference(parameters.means, parameters.log_scales, parameters.quaternions, parameters.opacity_logits, parameters.sh_coefficients, camera)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(args.output, image.detach().cpu().numpy())
        print(f"Rendered differentiable reference image on {device} to {args.output}")
    elif args.command == "train-colmap":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_sparse_model(args.sparse_model)
        initial_map = initialise_from_sparse_model(model)
        if len(initial_map) > args.max_gaussians:
            selection = np.random.default_rng(7).choice(len(initial_map), args.max_gaussians, replace=False)
            initial_map = GaussianMap(initial_map.means[selection], initial_map.log_scales[selection], initial_map.quaternions[selection], initial_map.opacity_logits[selection], initial_map.sh_coefficients[selection])
        initial_map = initial_map.with_sh_degree(args.sh_degree)
        views = load_training_views(model, args.image_directory, device, args.max_resolution)
        trainer = GaussianTrainer(initial_map, views, TrainingConfig(iterations=args.iterations, renderer=args.renderer), device)
        trained_map = trainer.train()
        trained_map.save_npz(args.output)
        history_path = args.output.with_suffix(".history.json")
        history_path.write_text(json.dumps(trainer.loss_history), encoding="utf-8")
        print(f"Trained {len(trained_map)} Gaussians for {args.iterations} iterations on {device}; saved {args.output}")
    elif args.command == "evaluate-colmap":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_sparse_model(args.sparse_model)
        views = load_training_views(model, args.image_directory, device, args.max_resolution)
        parameters = TorchGaussianParameters(GaussianMap.load_npz(args.gaussian_map), device)
        scores = evaluate_views(parameters, views, args.renderer)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"PSNR {scores['psnr']:.2f}, SSIM {scores['ssim']:.4f}; saved {args.output}")
    else:
        GaussianMap.load_npz(args.input).export_ply(args.output)
        print(f"Exported {args.output}")
