# NNDL Project 2 — CIFAR-10 Custom CNN (Task 1)

复旦大学《神经网络与深度学习》Project 2。当前仓库已完成 **Task 1**：在 CIFAR-10 上训练自定义 CNN（CIFARNet），并支持消融实验、评估与可视化。

Task 2（VGG + Batch Normalization）代码在 `codes/VGG_BatchNorm/`，仍为课程 starter，尚未完成全部实验。

## 项目结构

```
PJ2/
├── data/                          # CIFAR-10 数据集（不提交 GitHub）
├── codes/
│   ├── common/                    # Task 1 / Task 2 公用代码
│   │   ├── data/loaders.py        # CIFAR-10 数据加载
│   │   └── utils/
│   │       ├── device.py          # NPU / CUDA / CPU 设备检测与初始化
│   │       └── nn.py              # 权重初始化
│   ├── CIFAR10/                   # Task 1
│   │   ├── models/cnn.py          # CIFARNet 模型
│   │   ├── train.py               # 训练脚本
│   │   ├── evaluate.py            # 评估 checkpoint
│   │   ├── visualize.py           # 曲线 / 滤波器 / loss landscape
│   │   └── outputs/               # 训练输出（不提交 GitHub）
│   └── VGG_BatchNorm/             # Task 2 starter
└── run_task1_experiments.py       # 一键跑 Task 1 全部正式实验
```

## 环境依赖

建议使用独立 conda / venv 环境。

**通用依赖：**

```bash
pip install torch torchvision numpy matplotlib tqdm
```

**昇腾 NPU 环境（如 ModelArts / 华为云 Ascend 910）：**

除上述包外，还需安装 `torch_npu` 及 Ascend CANN toolkit（由平台镜像或官方文档提供）。代码已通过 `codes/common/utils/device.py` 适配 NPU，会自动：

- 禁用有问题的 `torch_npu` 自动加载
- 修复 `triton` 与 `torch_npu` 的版本兼容问题
- 按 **npu → cuda → cpu** 优先级选择默认设备

NPU 环境下建议 DataLoader 使用 `--num-workers 0`。

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

## 快速 Debug（验证流程）

```bash
cd codes/CIFAR10
python train.py --epochs 1 --n-items 256 --batch-size 64 --num-workers 0 --run-name debug
```

脚本会自动检测可用设备（NPU 环境下输出 `Device: npu`）。

## 训练单个实验

在 `codes/CIFAR10/` 下：

```bash
# 主实验（baseline，自动使用 NPU / CUDA / CPU）
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

训练结果保存在 `codes/CIFAR10/outputs/<run-name>/`：

- `best_model.pt` — 最佳权重
- `history.json` — 每 epoch 的 loss / accuracy
- `config.json` / `summary.json` — 超参与最终结果

## 一键跑全部 Task 1 正式实验

在项目根目录：

```bash
# 预览将要执行的命令
python run_task1_experiments.py --dry-run

# 跑全部 9 组实验（自动使用 NPU）
python run_task1_experiments.py --num-workers 0

# 只跑某一类消融
python run_task1_experiments.py --group width --num-workers 0
python run_task1_experiments.py --group loss --num-workers 0
python run_task1_experiments.py --group activation --num-workers 0
python run_task1_experiments.py --group optimizer --num-workers 0

# 跳过已有结果的实验（断点续跑）
python run_task1_experiments.py --num-workers 0 --skip-existing
```

实验组包括：baseline、width、loss、activation、optimizer。汇总报告写入 `codes/CIFAR10/outputs/experiments_report.json`。

## 评估与可视化

某次实验训练完成后，在 `codes/CIFAR10/` 下：

```bash
# 评估 test accuracy
python evaluate.py --checkpoint outputs/cifarnet/best_model.pt

# 生成训练曲线、第一层滤波器、loss landscape
python visualize.py --run-name cifarnet

# 跳过 loss landscape（更快）
python visualize.py --run-name cifarnet --skip-landscape
```

输出图片保存在对应 `outputs/<run-name>/` 目录下。

## CIFARNet 结构简述

- 4 个 Conv Block：Conv2d → BatchNorm → Activation → Conv2d → BatchNorm → Activation → MaxPool
- 分类头：Global Avg Pool → Dropout → FC → Dropout → FC(10)
- 详见 `codes/CIFAR10/models/cnn.py`
