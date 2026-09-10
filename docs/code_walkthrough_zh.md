# 项目代码拆解

阅读顺序建议是：`model.py` -> `colmap.py` -> `renderer.py` -> `trainer.py`。不要一开始就看所有文件；先沿着一张照片如何变成一次训练步骤来走。

## 数据与状态

`src/gsplat_scratch/model.py`

- `GaussianMap` 是可保存的 NumPy 模型文件。
- 它保存中心、对数尺度、四元数、opacity logit、SH 系数。
- `save_npz`/`load_npz` 用于工作文件；`export_ply` 用于交换和查看。
- `with_sh_degree(3)` 把每个颜色通道扩展到 16 个球谐系数。

`src/gsplat_scratch/torch_model.py`

- `TorchGaussianParameters` 把同一组数组包装成 `torch.nn.Parameter`。
- 这一步之后，PyTorch 才能对这些数字自动求导并让 Adam 更新它们。

## COLMAP 输入

`src/gsplat_scratch/colmap.py` 手写读取 `cameras.bin`、`images.bin` 和 `points3D.bin`，也支持文本版本。特别要注意 COLMAP 位姿是 world-to-camera：

```text
x_camera = R * x_world + t
```

`src/gsplat_scratch/initialization.py` 将稀疏点云的 RGB 变成零阶 SH 颜色，用最近邻距离变成初始尺度。`dataset.py` 读取训练照片，并在缩小分辨率时同步缩放焦距和主点。

## 渲染

`src/gsplat_scratch/renderer.py` 是项目的核心。

- `project_gaussians`：计算相机空间坐标、二维中心、二维协方差和 3-sigma 半径。
- `sh_colours`：按相机到高斯的观察方向计算 0 到 3 阶 SH 颜色。
- `render_reference`：每个高斯检查每个像素，最直白的正确性实现。
- `render_tiled`：把屏幕切成 16 x 16 tile，只在相交 tile 内计算该高斯；可将张量放进 CUDA。

`tests/test_renderer.py` 很重要。它验证中心投影、前后遮挡、自动求导、SH 视角相关性，以及 tiled 输出和 reference 输出一致。

## 训练与密度控制

`src/gsplat_scratch/trainer.py`

每一步随机取一张训练图：渲染 -> L1 与 DSSIM -> `backward()` -> Adam 更新。参数分组意味着位置、颜色、透明度、尺度和旋转各自用适合的学习率。

`src/gsplat_scratch/density.py`

`DensityController` 累积二维中心梯度：小而难拟合的高斯被 clone，大而难拟合的高斯被 split，低 opacity 的高斯被 prune。参数数量变化时，训练器把当前状态转回 `GaussianMap`，重建参数和优化器；这牺牲了一部分优化器动量，但逻辑透明，适合作为从零复现的教学版本。

## 指标、查看器与命令行

- `metrics.py`：PSNR 和 SSIM。
- `viewer.py`：Matplotlib 椭球查看器，可以选中、移动、缩放、改 opacity 并保存。
- `cli.py`：把所有能力组合为 `demo`、`view`、`import-colmap`、`train-colmap`、`evaluate-colmap` 等命令。

## 测试应该如何读

`tests/` 不是附属品。每个测试都在固定一个容易写错的数学事实：四元数方向、COLMAP 二进制布局、最近邻尺度、alpha 遮挡、梯度存在、split 替换父节点、训练损失下降。看不懂某个模块时，先读同名测试，往往比从实现细节硬读更快。
