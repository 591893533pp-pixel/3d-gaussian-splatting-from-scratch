# 3D Gaussian Splatting From Scratch

从零复现 Kerbl et al. 的 *3D Gaussian Splatting for Real-Time Radiance Field Rendering*。

本项目不会调用官方实现中的训练代码或 CUDA rasterizer。允许使用 PyTorch 自动求导、COLMAP 相机标定结果与 CUDA/Triton；3DGS 的参数表示、投影、alpha 合成、密度控制和渲染器均自行实现。

## 当前阶段

第一阶段已建立可编辑高斯地图基础：

- `GaussianMap` 保存高斯中心、对数尺度、旋转四元数、opacity logits 与零阶球谐系数。
- 使用 `.npz` 进行无损工作文件保存，使用 ASCII `.ply` 进行交换导出。
- `matplotlib` 编辑器以椭球显示各向异性协方差；可选中单个高斯并实时调节位置、尺度与不透明度。

## 环境

现有 Conda 环境为 `3dgs`，已验证 PyTorch 能识别 CUDA 12.8 和 RTX 5080。

```powershell
conda activate 3dgs
pip install -e .
```

## 运行地图编辑器

先创建可编辑演示地图：

```powershell
python -m gsplat_scratch demo outputs/demo_map.npz --count 48
python -m gsplat_scratch view outputs/demo_map.npz --output outputs/demo_map_edited.npz
```

编辑器窗口中可用鼠标旋转、缩放和移动视角。下方的滑块用于选择高斯并修改参数；`Save map` 会写入 `--output` 指定的文件。

## 导入 COLMAP 稀疏重建

将 COLMAP 的 `sparse/0` 目录传入命令。支持 COLMAP 的 `cameras.bin`、`images.bin`、`points3D.bin`，也支持对应的文本格式。每个稀疏点会以其 RGB 颜色和最近邻距离初始化为一个各向异性高斯。

```powershell
python -m gsplat_scratch import-colmap path/to/sparse/0 outputs/colmap_map.npz
python -m gsplat_scratch view outputs/colmap_map.npz --output outputs/colmap_map_edited.npz
```

## 可微投影与参考渲染

`renderer.py` 是为验证公式而写的 PyTorch 参考 rasterizer：它将旋转的 3D 协方差用透视 Jacobian 投影为 2D 椭圆，并按深度前到后 alpha 合成。它的复杂度为 `O(NHW)`，只用于小分辨率正确性验证；后续 CUDA tile rasterizer 将替换这一性能瓶颈。

```powershell
python -m gsplat_scratch render-demo outputs/reference_render.png
```

## 参考训练循环

训练器使用 L1 + DSSIM、分组 Adam、二维投影梯度统计和 clone/split/prune 密度控制。默认 `tiled` 后端会在 CUDA 上按屏幕 tile 筛选高斯，避开参考后端的全局 `O(NHW)` 张量；它是便于审查的 PyTorch GPU 实现，尚不是论文级的自定义 CUDA kernel。

```powershell
python -m gsplat_scratch train-colmap path/to/sparse/0 path/to/images outputs/trained_map.npz --iterations 500 --max-resolution 128 --max-gaussians 256 --sh-degree 3
python -m gsplat_scratch evaluate-colmap outputs/trained_map.npz path/to/sparse/0 path/to/images outputs/metrics.json
```

## 路线图

1. 高斯地图、序列化与可视编辑器。
2. COLMAP 读取、相机模型、稀疏点云初始化。
3. PyTorch 可微投影与 alpha 合成正确性版本。
4. 优化、SH 外观、增密、分裂、裁剪与评测。
5. CUDA/Triton tile rasterizer 与实时渲染。

## 参考

- [论文](https://arxiv.org/abs/2308.04079)
- [作者项目页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## 学习资料

- [论文拆解](docs/paper_walkthrough_zh.md)
- [代码拆解](docs/code_walkthrough_zh.md)
- [从零使用与数据集建议](docs/usage_and_datasets_zh.md)

