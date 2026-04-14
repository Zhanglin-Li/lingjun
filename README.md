# Lingjun 项目文档

## 项目概览

灵均 A 股微观结构预测比赛。目标是预测 500 只股票在每个时间步上的 LabelA（主目标），数据中另有 LabelB 和 LabelC 两个辅助标签（当前未使用）。

**数据特点：**
- 原始数据 parquet 格式，约 4300 万行、42GB
- 每条记录包含 `stockid`（股票 id）、`dateid`（日期id）、`timeid`（日内时间id 0-238），以及 384 个特征 `f0`-`f383`
- exchange：交易所编号，取值为 0 或 1
- 比赛要求提交源码，禁止使用未来数据

**提交格式：** `stockid|dateid|timeid,prediction`

## 数据预处理

```
原始 parquet → 截面 zscore 标准化 → NaN 填 0 → 按日期打包 → DataLoader
```

### 标准化

对每个 `(dateid, timeid)` 截面，将 384 个特征分别做 zscore 标准化（减去截面均值、除以截面标准差），消除不同时间截面上的量纲和分布差异。

### NaN 处理

原始数据中部分特征存在缺失，标准化过程（除以标准差为 0 的情况）也会产生 NaN，统一填充为 0。

注意 Polars 中 `fill_nan` 处理浮点 NaN，`fill_null` 处理空值，两者需先后调用。

### 数据集组织

数据按 `dateid`（天）为单位打包。训练时每个 sample 是一整天的数据，形状为 `(n_stocks, 239, n_features)`。

**Train/Val 划分**：时间序列上按天切分，前 300 天训练，后 60 天验证。

## 模型架构 (ModelR)

GRU 序列模型，输入一整天的特征序列，输出每个时间步的 LabelA 预测：

```
Input (stocks, 239, features)
         |
   ┌─────────────┐
   │  GRU Layer  │  (hidden_size 可配置，如 512)
   │  + Dropout  │
   └──────┬──────┘
          │
   ┌─────────────┐
   │  GRU Layer  │  (可多层，如 256)
   │  + Dropout  │
   └──────┬──────┘
          │
   ┌─────────────┐
   │  FC Head    │  Linear → (stocks, 239, 1)
   └─────────────┘
          │
    → LabelA pred
```

- 多层 GRU 逐层提取时序特征，每层后接 Dropout
- 最后一层 GRU 输出通过一个 Linear 层映射为标量预测
- 输出形状 `(stocks, 239)`，即每只股票每个时间步一个预测值

## Loss

 R² Loss：`R² = 1 - SS_res / SS_tot`


## 训练流程

### 配置

配置文件在 `config/` 目录下，YAML 格式。主要参数：学习率、batch size、GRU hidden sizes、early stopping patience 等。

### 运行方式

- **`train.py`**：主训练脚本，接受 config 路径作为第一个参数
- **`train.ipynb`**：Jupyter notebook，与 `train.py` 相同的 HuggingFace Trainer 流程

### HuggingFace Trainer 细节

- 按 step 评估和保存：每 100 step 在验证集上计算 R²
- Early stopping 基于验证集 R²（`metric_for_best_model='eval_r2'`）
- `load_best_model_at_end=True` 要求 `save_strategy == eval_strategy`
- Dataset 的 `__len__` 返回 `actual_dates × repeat`，`__getitem__` 通过取模循环遍历日期，`repeat` 控制总 epoch 数

### Checkpoint

模型保存文件名包含实验编号、时间戳和验证集 R²。当验证 R² 提升时替换旧 checkpoint。

## Online Learning

验证阶段引入在线学习机制，参考 Jane Street 方案：

验证时对模型的一个**深拷贝副本**进行逐 batch 的在线更新，仅用 LabelA 的 Loss。更新后的模型继续用于后续 batch 的预测。

**为什么要深拷贝：** 防止验证集信息泄露回训练模型。

**关键参数：**
- 每 batch 新建独立的 AdamW 优化器（不累积 momentum）
- 学习率 0.00006，weight_decay 0.01
- 梯度裁剪 max_norm=1.0

这让模型在验证时能适应数据分布的漂移，更贴近真实推理场景。

## 项目文件结构

```
├── config/                  # YAML 配置文件
├── model.py                 # 模型定义（ModelR, WeightedR2Loss）
├── train.py                 # 主训练脚本（HuggingFace Trainer）
├── train.ipynb              # Jupyter notebook 版训练
├── run_sweep.sh             # 批量实验脚本
├── data/                    # 原始数据（parquet）
├── experiments/             # 实验记录 CSV
└── checkpoints/             # 模型权重
```

## 与 Jane Street 方案的对应关系

本项目架构从 Jane Street 方案迁移而来，核心区别：

| | Jane Street | Lingjun |
|---|---|---|
| 日内时间步 | ~968 | 239 |
| 特征数 | ~80 | 384 |
| 实体 | `symbol_id` | `stockid` (0-499) |
| 交易所 | 无 | 两个交易所，分布不同 |
| 在线学习 | 推理阶段可用 | 受限于源码提交要求 |

## 快速上手

```bash
# 激活环境 & 安装依赖
source .venv/bin/activate && uv sync

# 运行训练（默认配置）
.venv/bin/python train.py config/template.yaml

# 检查语法
.venv/bin/python -m py_compile train.py
```
