# Lingjun China A-Share Market Microstructure Prediction

A股市场微观结构预测竞赛项目，基于高频交易数据预测股票日内收益率。

## 项目背景

### 竞赛概述

- **目标**: 预测中国A股市场500只股票的日内收益率 (`LabelA`)
- **数据**: 数百天的高频L2市场数据，包含384个匿名特征 (`f0`-`f383`)
- **评估指标**: R² (决定系数)
- **数据规模**: ~4300万行，42GB

### 数据特点

- 500只股票 (`stockid` 0-499)，分属两个交易所 (`exchangeid`)
- 每日239个时间步 (`timeid` 0-238)
- 辅助标签: `LabelB`, `LabelC` (不同时间窗口的收益率)
- 最后10个时间步 (229-238) 不参与评分但需提交

## 方法（baseline ：把 janestreet 方法直接套过来）

### 数据处理

```python
# 1. 特征选择: 按重要性筛选Top 80特征
# 2. 滚动统计 (窗口=239，即一个完整交易日， 原js approach 大致1d)
pl.col(feature).rolling_mean(239).over('stockid')  # 滚动均值
pl.col(feature).rolling_std(239).over('stockid')   # 滚动标准差

# 3. 截面统计 (每时刻所有股票的均值)
pl.col(feature).mean().over('dateid', 'timeid')

# 4. 标准化与缺失值处理（original js approach point out that other method won't help)
df = df.with_columns([
    pl.col(col).fill_nan(0.0).fill_null(0.0) for col in feature_cols
])
```

**特征工程后总计**: ~129个特征 (80原始 + 滚动统计 + 截面统计)

### 模型架构

采用 GRU 时序模型:

```
Input (batch, 239, features)
    ↓
GRU Layer 1 (hidden=512)
    ↓ Dropout(0.1)
GRU Layer 2 (hidden=256)
    ↓ Dropout(0.1)
Linear (500)
    ↓ Dropout(0.2)
Linear (300)
    ↓
Output (batch, 239, 1)
```

**多任务学习**: 同时预测 `LabelA` (主目标)、`LabelB`、`LabelC` 
通过画图和其他一些辅助措施基本可确认LabelA/B/C 分别是 T+1~T+10/T+1~T+30/T+1~T+60 平均收益率（注意这里 T 不是dateid，是时间步 timeid）
详见 js 中提到的逆向数据工程方法

模型架构（参考原 js）：
k 个辅助任务+1 个主任务，用 k 个 rnn 分别 train k 个辅助任务（第 i 个辅助任务的loss 是batch 上的 R2），最后把 k 个预测值接成 k 维向量再过一个线性层得到主任务预测值。
略诡异，可能后需要调整？

### 训练策略

**Online Learning** (参考Jane Street方案):

- 每个 batch 后执行一次梯度更新
- 优化器: AdamW (lr=0.00006, weight_decay=0.01)
- 梯度裁剪: max_norm=1.0

**训练参数**:

- Batch size: 4 (按dateid分组，每batch约500只股票×239时间步)
- 学习率: 0.0001 (主训练), 0.00006 (在线学习)
- Early stopping: patience=10
- 训练/验证划分: 288天/72天

### 损失函数

加权 R² 损失:

```python
def weighted_r2_loss(y_pred, y_true, weights):
    y_mean = (weights * y_true).sum() / weights.sum()
    ss_res = (weights * (y_pred - y_true) ** 2).sum()
    ss_tot = (weights * (y_true - y_mean) ** 2).sum()
    return ss_res / (ss_tot + 1e-38)
```

## 当前成果

### 最佳结果


| 配置      | 数据集                      | 模型            | 验证R²        |
| ------- | ------------------------ | ------------- | ----------- |
| exp_012 | filtered.parquet (224股票) | GRU [512,256] | **+0.0102** |
| exp_016 | train.parquet (500股票)    | GRU [512,256] | **+0.0077** |


### 关键发现

1. **学习率**: 0.0001 最优，更高学习率导致负R²
2. **模型容量**: 两层GRU [512, 256] 优于单层 (欠拟合) 和更大容量 (过拟合)
3. **Dropout**: 0.1-0.3 效果好，0.5 过于激进
4. **特征数量**: 80个精选特征最优，更多特征带来噪声

## 项目结构

```
├── config/                  # 配置文件
├── data/                    # 数据文件 (train.parquet, test.parquet)
├── experiments/             # 实验日志
├── checkpoints/             # 模型检查点
├── docs/                    # 文档
├── run_training_simple.py   # 训练脚本
├── model.py                 # 模型定义
└── custom_dataset.py        # 数据加载
```

## 使用方法

```bash
# 激活环境
source .venv/bin/activate

# 安装依赖
uv sync

# 训练模型
python run_training_simple.py config/config.yaml

# 查看实验日志
cat experiments/experiment_log.csv
```

## 参考

- [Jane Street Market Prediction](https://www.kaggle.com/c/jane-street-market-prediction) - 在线学习策略参考

