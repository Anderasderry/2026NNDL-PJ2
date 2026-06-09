# NNDL Project 2 — CIFAR-10

复旦大学《神经网络与深度学习》Project 2。

- **Task 1**：在 CIFAR-10 上训练自定义 CNN（CIFARNet），支持网格搜索、评估与可视化。
- **Task 2**：VGG-A 与 VGG-A+BatchNorm 对比，以及 Loss / Gradient 优化分析。

部分实验结果（图表、JSON）已包含在 `outputs/` 中，可直接查阅或用于重绘。

## 实验结果摘要

| 任务 | 最佳配置 | 主要指标 |
|------|----------|----------|
| Task 1 | `width96_final`（width=96，wd=5e-4，200 epochs） | Test accuracy **94.23%**（error 5.77%） |
| Task 2 | VGG-A+BN vs VGG-A（Adam，lr=1e-3，20 epochs） | Val accuracy **87.01%** vs 82.90%（+4.11 pp） |

## 项目结构

```
PJ2/
├── data/                          # CIFAR-10 数据集
├── logs/                          # 实验日志
├── weights/                       # 预训练权重（需从网盘下载）
├── outputs/                       # 训练输出
│   ├── CIFAR10/                   # Task 1
│   └── VGG_BatchNorm/             # Task 2：figures/、models/、vgg_a/ 等
├── codes/
│   ├── common/                    # Task 1 / Task 2 公用代码
│   │   ├── paths.py               # 数据 / 日志 / 输出路径
│   │   ├── data/loaders.py        # CIFAR-10 数据加载
│   │   └── utils/
│   │       ├── device.py          # NPU / CUDA / CPU 设备检测与初始化
│   │       └── nn.py              # 权重初始化
│   ├── CIFAR10/                   # Task 1
│   │   ├── models/cnn.py          # CIFARNet 模型
│   │   ├── train.py               # 训练脚本
│   │   ├── evaluate.py            # 评估 checkpoint
│   │   └── visualize.py           # 曲线 / 滤波器 / loss landscape
│   └── VGG_BatchNorm/             # Task 2
│       ├── models/vgg.py          # VGG_A / VGG_A_BatchNorm
│       ├── VGG_Loss_Landscape.py  # 课程 starter + CLI 入口
│       ├── experiments.py         # 完整实验流程与 argparse
│       ├── train_VGG.py           # 训练 / 评估工具
│       ├── core.py                # landscape 曲线计算
│       └── plots.py               # landscape 绘图
├── run_task1.sh                   # 一键跑 Task 1 网格搜索（50 epochs）
└── run_task2.sh                   # 一键跑 Task 2 训练 + landscape + 重绘图
```

## 环境依赖

建议使用独立 conda / venv 环境。

**通用依赖：**

```bash
pip install torch torchvision numpy matplotlib tqdm
```

代码已通过 `codes/common/utils/device.py` 适配 NPU，会自动按 **npu → cuda → cpu** 优先级选择默认设备。

## 准备数据集

数据目录固定为项目根目录下的 `data/`：

```
PJ2/data/cifar-10-batches-py/
```

或放置完整压缩包（约 170 MB），首次训练时会自动解压：

```
PJ2/data/cifar-10-python.tar.gz
```

也可手动下载：<https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

## 预训练权重

`*.pt` 权重文件不入库。从 [Google Drive](https://drive.google.com/drive/folders/19UBlpJP-q3F3ViugIzdemxbVWqC5CKnO?usp=sharing) 下载后放入项目根目录 `weights/`：

| 文件 | 说明 |
|------|------|
| `task1_best_model.pt` | CIFARNet 最佳模型（对应 `width96_final`） |
| `vgg_a.pt` | Task 2 VGG-A |
| `vgg_a_bn.pt` | Task 2 VGG-A+BN |

下载后可直接评估：

```bash
# Task 1
cd codes/CIFAR10
python evaluate.py --checkpoint ../../weights/task1_best_model.pt
python visualize.py --checkpoint ../../weights/task1_best_model.pt --run-name width96_weights --skip-landscape --num-workers 0
```

Task 2 训练产出的权重默认在 `outputs/VGG_BatchNorm/models/`；若使用网盘文件，可复制到 `weights/` 或 `outputs/VGG_BatchNorm/models/` 后加载。

## 快速 Debug（验证流程）

```bash
cd codes/CIFAR10
python train.py --epochs 1 --n-items 256 --batch-size 64 --num-workers 0 --run-name debug
```

## 训练单个实验

在 `codes/CIFAR10/` 下：

```bash
# 主实验（baseline）
python train.py --epochs 200 --run-name cifarnet

# 常用可调参数
python train.py --width 64 --activation relu --loss ce --optimizer sgd --lr 0.1 --run-name my_run

# 显式指定设备
python train.py --device npu --epochs 200 --num-workers 0 --run-name cifarnet
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 200 |
| `--batch-size` | batch 大小 | 128 |
| `--width` | 基础通道数（filter 数） | 64 |
| `--activation` | `relu` / `gelu` / `leaky_relu` | relu |
| `--loss` | `ce` / `label_smooth` | ce |
| `--optimizer` | `sgd` / `adamw` | sgd |
| `--lr` | 学习率 | 0.1（AdamW 建议 3e-4） |
| `--weight-decay` | L2 正则 | 5e-4 |
| `--run-name` | 输出子目录名 | cifarnet |
| `--device` | `npu` / `cuda` / `cpu` | 自动检测（npu → cuda → cpu） |
| `--num-workers` | DataLoader 进程数 | 2（NPU 建议 0） |

训练结果保存在 `outputs/CIFAR10/<run-name>/`：

- `best_model.pt` — 最佳权重
- `history.json` — 每 epoch 的 loss / accuracy
- `config.json` / `summary.json` — 超参与最终结果

## 一键跑 Task 1 网格搜索

在项目根目录：

```bash
# 9 组网格搜索（50 epochs）+ 汇总 experiments_report.json
bash run_task1.sh

# 跳过已有 summary.json 的实验（断点续跑，默认开启）
bash run_task1.sh

# 强制全部重跑
SKIP_EXISTING=0 bash run_task1.sh
```

`run_task1.sh` 依次执行：

1. 9 组网格搜索（baseline / width / loss / activation / optimizer，各 50 epochs）
2. 汇总写入 `outputs/CIFAR10/experiments_report.json`

**选定最优配置后**，再单独跑 200 epoch 正式训练，例如：

```bash
cd codes/CIFAR10
python train.py --epochs 200 --width 96 --run-name width96_final --num-workers 0
python evaluate.py --checkpoint ../../outputs/CIFAR10/width96_final/best_model.pt
python visualize.py --run-name width96_final --skip-landscape
```

## 评估与可视化

某次实验训练完成后，在 `codes/CIFAR10/` 下：

```bash
# 评估 test accuracy（路径相对于 codes/CIFAR10/）
python evaluate.py --checkpoint ../../outputs/CIFAR10/width96_final/best_model.pt

# 生成训练曲线、第一层卷积滤波器；可选 1D loss landscape
python visualize.py --run-name width96_final

# 跳过 loss landscape（更快）
python visualize.py --run-name width96_final --skip-landscape
```

`visualize.py` 默认输出 `training_curves.png`、`first_conv_filters.png`、`confusion_matrix.png`、`misclassified_examples.png`；加 `--skip-interpretation` 可跳过后两者；未加 `--skip-landscape` 时额外生成 `loss_landscape.png`。图片保存在 `outputs/CIFAR10/<run-name>/`。

## CIFARNet 结构简述

- 4 个 Conv Block：Conv2d → BatchNorm → Activation → Conv2d → BatchNorm → Activation → MaxPool
- 分类头：Global Avg Pool → Dropout → FC → Dropout → FC(10)
- 详见 `codes/CIFAR10/models/cnn.py`

---

## Task 2：VGG-A + Batch Normalization

### 目标

1. **VGG-A vs VGG-A+BN**：对比有无 BatchNorm 的训练效果与收敛特性。
2. **优化分析**：通过 Loss Landscape、Gradient Predictiveness（\|g_t - g_{t-1}\|）、Max Gradient Difference（跨学习率 max−min）分析 BN 如何平滑优化过程。

### 模型

| 模型 | 文件 | 说明 |
|------|------|------|
| `VGG_A` | `codes/VGG_BatchNorm/models/vgg.py` | 原始 VGG-A（32×32 输入） |
| `VGG_A_BatchNorm` | 同上 | 每个 Conv 后加 `BatchNorm2d` |

### 运行实验

**一键跑完整流程**（项目根目录）：

```bash
# 训练 + landscape + 对比图重绘
bash run_task2.sh

# 快速调试
EPOCHS=2 N_ITEMS=1024 bash run_task2.sh

# 已有 JSON，只重绘对比图
SKIP_TRAIN=1 bash run_task2.sh
```

也可在 `codes/VGG_BatchNorm/` 下手动运行。**带 CLI 参数**时由 `VGG_Loss_Landscape.py` 转发到 `experiments.py`；**无参数**时运行课程 starter 脚本（需 IPython）：

```bash
# 完整实验：对比训练 + loss/gradient landscape（默认 20 epochs）
python VGG_Loss_Landscape.py --num-workers 0

# 快速调试
python VGG_Loss_Landscape.py --epochs 2 --n-items 1024 --num-workers 0

# 只跑 VGG-A vs BN 对比（跳过 landscape）
python VGG_Loss_Landscape.py --skip-landscape --epochs 20 --num-workers 0

# 只跑 landscape 分析（跳过单次对比）
python VGG_Loss_Landscape.py --skip-comparison --epochs 20 --num-workers 0

# 自定义学习率（用于 landscape 实验）
python VGG_Loss_Landscape.py --learning-rates 1e-3 2e-3 1e-4 5e-4 --num-workers 0

# 仅从已有 JSON 重绘两张对比图（无需 GPU / 重新训练）
python VGG_Loss_Landscape.py --replot-comparison \
  --loss-ylim 0 2.5 --predictiveness-ylim 0 1 \
  --plot-stride 50 --fill-alpha 0.15

# 只重绘 loss landscape 对比图
python VGG_Loss_Landscape.py --replot-loss-landscape \
  --loss-ylim 0 2.5 --plot-stride 50 --fill-alpha 0.15

# 只重绘 gradient predictiveness 对比图
python VGG_Loss_Landscape.py --replot-predictiveness --plot-stride 1
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 20 |
| `--batch-size` | batch 大小 | 128 |
| `--lr` | 对比实验的学习率 | 1e-3 |
| `--learning-rates` | landscape 用的多组 lr | 1e-3 2e-3 1e-4 5e-4 |
| `--n-items` | 子集大小（-1 为全量） | -1 |
| `--num-workers` | DataLoader 进程数 | 0 |
| `--skip-comparison` | 跳过 VGG-A vs BN 对比 | — |
| `--skip-landscape` | 跳过 landscape 实验 | — |
| `--replot-comparison` | 重绘上述两张对比图 | — |
| `--replot-loss-landscape` | 仅重绘 `loss_landscape_comparison.png` | — |
| `--replot-predictiveness` | 仅重绘 `grad_predictiveness_comparison.png` | — |
| `--loss-ylim` | loss landscape 对比图 y 轴范围 | 自动 |
| `--predictiveness-ylim` | predictiveness 对比图 y 轴范围 | 自动 |
| `--plot-stride` | 对比图下采样步长 | 50 |
| `--fill-alpha` | landscape 填充透明度 | 0.15 |

输出目录为 `outputs/VGG_BatchNorm/`（由 `common/paths.py` 统一管理）。

### 输出文件

结果保存在 `outputs/VGG_BatchNorm/`：

**训练对比：**

| 文件 | 内容 |
|------|------|
| `figures/vgg_a_training_curve.png` | VGG-A 训练 loss / 验证准确率 |
| `figures/vgg_a_bn_training_curve.png` | VGG-A+BN 训练曲线 |
| `vgg_a/summary.json` | VGG-A 最佳验证准确率等 |
| `vgg_a_bn/summary.json` | VGG-A+BN 结果摘要 |
| `models/vgg_a.pt` / `models/vgg_a_bn.pt` | 最佳模型权重（`*.pt` 不入库时可从网盘获取） |

**优化分析（Loss / Gradient）：**

| 文件 | 内容 |
|------|------|
| `figures/loss_landscape_comparison.png` | 两者 loss landscape 同图对比 |
| `figures/grad_predictiveness_comparison.png` | Gradient predictiveness：\|g_t - g_{t-1}\| 逐步变化量 |
| `figures/vgg_a_grad_max_diff.png` | VGG-A 每 step 跨 lr 的 max(grad)−min(grad) |
| `figures/vgg_a_bn_grad_max_diff.png` | VGG-A+BN 同上 |
| `vgg_a_landscape.json` / `vgg_a_bn_landscape.json` | landscape 数值数据（可 `--replot-*` 重绘） |