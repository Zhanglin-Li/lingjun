# 实验日志与调参方案

## Context

当前训练模型存在以下问题：
- R² 始终为负（最佳约 -0.006）
- 模型收敛慢，可能欠拟合
- 缺少实验记录，无法对比不同配置

需要：
1. 添加实验日志系统，记录超参和结果
2. 制定系统的调参方案，逐参数调试

---

## Part 1: 实验日志系统 ✅ 已完成

在 `run_training_simple.py` 中添加了 `ExperimentLogger` 类，自动记录到 `experiments/experiment_log.csv`。

---

## Part 2: 调参方案

### 调参原则
- **一次只调一个参数**，其他保持默认
- 每次实验记录完整配置和结果
- 基于前一次结果决定下一步

### 基准配置 (Baseline - 已更新为最优)

```yaml
model:
  type: gru
  hidden_sizes: [512, 256]
  dropout_rates: [0.1, 0.1, 0.0]
  hidden_sizes_linear: [500, 300]
  dropout_rates_linear: [0.2, 0.1]

training:
  batch_size: 4  # 固定，GPUDataset 内存限制
  lr: 0.0001  # 已优化
  early_stopping_patience: 10  # 待调试

features:
  cols_init: 前 80 个特征
  cols_corr: 前 16 个特征（用于 rolling/cross-sectional，= 80/5）
```

---

### 调参顺序

#### Phase 1: 特征数量 ✅ 已完成

**结论**：Baseline 80 特征最优，更多特征反而降低性能。

---

#### Phase 2: 学习率 ✅ 已完成

**结论**：lr=0.0001 最优，达到 +0.0032 R² (BS=4)。

---

#### Phase 3: 模型容量 ✅ 已完成

**结论**：两层 GRU [512, 256] 最优，达到 +0.0102 R²，比单层 [500] 提升 3 倍。

---

#### Phase 4: Dropout (正则化) ✅ 已完成

**结论**：Dropout 0.1-0.3 均可，0.5 过强导致性能下降。

---

#### Phase 5: 早停耐心值 (待运行)

配置文件已创建：`config_patience15.yaml`, `config_patience20.yaml`

| 实验号 | 参数 | 值 | 说明 |
|--------|------|-----|------|
| 14 | early_stopping_patience | 15 | 允许更长训练 |
| 15 | early_stopping_patience | 20 | 更长训练 |

**注意**：最小值为 10，已作为基准

---

#### Phase 6: 其他探索 (可选)

| 实验号 | 参数 | 值 | 说明 |
|--------|------|-----|------|
| 16 | model.type | lstm | LSTM vs GRU |
| 17 | weight_decay | 0.001 | 减小权重衰减 |
| 18 | weight_decay | 0.1 | 增大权重衰减 |

---

## 实现步骤

### Step 1: 修改 config.yaml 进行特征数量实验

```yaml
# 实验 1: 前 120 特征
features:
  cols_init:
    - f298
    - f285
    # ... 添加到前 120 个
```

### Step 2: 运行训练并记录

```bash
python run_training_simple.py
```

### Step 3: 查看实验日志

```bash
cat experiments/experiment_log.csv
```

---

## 关键文件

- `run_training_simple.py` - 训练脚本（已添加日志功能）
- `config.yaml` - 配置文件（手动修改参数）
- `experiments/experiment_log.csv` - 实验日志（自动生成）

---

## 注意事项

1. **删除缓存**：修改特征数量后需要删除缓存文件
   ```bash
   rm -f ./data/cache/train_standardized.parquet
   ```

2. **特征来源**：特征列表按重要性排序，前 80/120/160 个特征