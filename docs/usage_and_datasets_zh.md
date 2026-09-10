# 使用指南与数据集建议

## 0. 首次准备

在项目目录中打开 Anaconda PowerShell：

```powershell
conda activate 3dgs
pip install -e .
python -m unittest discover -s tests -v
```

看到全部测试通过，说明环境、依赖和项目导入正常。

## 1. 先跑无数据集演示

```powershell
python -m gsplat_scratch demo outputs/demo_map.npz --count 48
python -m gsplat_scratch view outputs/demo_map.npz --output outputs/demo_map_edited.npz
python -m gsplat_scratch render-demo outputs/reference_render.png
```

编辑窗口中：鼠标控制视角；`Gaussian` 滑条选择编号；`X/Y/Z` 移动中心；`log scale` 改三个轴长度；`opacity logit` 改透明度；`Save map` 保存修改。

## 2. 准备自己的场景

第一组真实数据建议自己拍一个静态小场景：书桌一角、盆栽、摆件或房间局部都合适。

- 拍 80 到 150 张图，围绕物体缓慢移动。
- 相邻照片保持约 60% 到 80% 重叠。
- 固定焦距和曝光；不要拍运动物体、反光强烈的镜子或纯白墙。
- 把照片放到 `my_scene/images/`。

用 COLMAP 做 feature extraction、matching 和 mapper 后，你应有：

```text
my_scene/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

## 3. 导入、训练、查看与评估

先只导入并查看稀疏点初始化：

```powershell
python -m gsplat_scratch import-colmap my_scene/sparse/0 outputs/my_scene_initial.npz
python -m gsplat_scratch view outputs/my_scene_initial.npz --output outputs/my_scene_initial_edited.npz
```

再以保守配置训练：

```powershell
python -m gsplat_scratch train-colmap my_scene/sparse/0 my_scene/images outputs/my_scene_trained.npz --iterations 500 --max-resolution 128 --max-gaussians 256 --sh-degree 3
python -m gsplat_scratch evaluate-colmap outputs/my_scene_trained.npz my_scene/sparse/0 my_scene/images outputs/my_scene_metrics.json
python -m gsplat_scratch view outputs/my_scene_trained.npz --output outputs/my_scene_edited.npz
```

`outputs/` 和 `data/` 被 Git 忽略，避免把大型照片和训练产物误提交。代码与文档应提交；实验参数、指标、数据来源可写进 README 或实验记录。

## 4. 推荐数据集

推荐顺序如下：

1. 自己拍的小型静态场景：最适合理解 COLMAP、相机位姿和编辑器，也没有下载成本。
2. Mip-NeRF 360 的 `bonsai` 或 `room`：这是论文使用的标准真实场景类别，适合验证室内细节；数据和处理说明可从 [MultiNeRF 官方仓库](https://github.com/google-research/multinerf) 获取。
3. Tanks and Temples 的 `Truck` 或 `Train`：论文评估使用过该基准的场景，官方提供图像集、视频和下载工具；先使用其较小图像集。遵守其非商业研究/教学许可：[官方下载页](https://tanksandtemples.org/download/)。
4. Deep Blending 的 `Playroom`：包含复杂室内遮挡与材质，适合在前述流程稳定后再挑战。请从 3DGS 作者项目页提供的版本取得与论文一致的预处理数据。

本项目当前的 tiled 后端适合低分辨率、小规模高斯的学习与验证。若要在完整 Mip-NeRF 360 场景上复现论文的训练时间和实时帧率，需要继续把 `render_tiled` 改写为 fused 自定义 CUDA 前向/反向 kernel，并扩展完整的测试集划分与 LPIPS 评估。
