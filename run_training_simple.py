#!/usr/bin/env python3
"""简化版训练脚本 - 只依赖基础包 (polars/numpy/tqdm/torch)

所有配置硬编码在脚本顶部，不依赖外部配置文件。
"""
import gc
import os
import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model import ModelR, WeightedR2Loss, r2_weighted_torch


# ============================================================
# 配置部分（硬编码，不依赖 config.yaml）
# ============================================================

# 数据路径
PATH_DATA = "./data"
PATH_PARQUET = os.path.join(PATH_DATA, "train.parquet")

# 列名配置
COL_TARGET = "LabelA"
COL_ID = "stockid"
COL_DATE = "dateid"
COL_TIME = "timeid"
COL_WEIGHT = None  # 你的数据没有 weight 列
COLS_RESPONDERS = ["LabelB", "LabelC"]

# 模型配置
MODEL_TYPE = "gru"
HIDDEN_SIZES = [500]
DROPOUT_RATES = [0.3, 0.0, 0.0]
HIDDEN_SIZES_LINEAR = [500, 300]
DROPOUT_RATES_LINEAR = [0.2, 0.1]

# 训练配置
BATCH_SIZE = 16
LR = 0.0005
EPOCHS_PER_SEED = {0: 8, 1: 8, 2: 8}  # 每个种子的 epochs
SEED_RANGE = range(1)  # 训练的种子范围
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 1
CHECKPOINT_INTERVAL = 5

# 数据划分
TRAIN_DAYS = 288  # 训练集天数
VALID_DAYS = 72   # 验证集天数
TOTAL_DAYS = TRAIN_DAYS + VALID_DAYS

# 特征配置（与 data_processor.py 保持一致）
COLS_INIT = [f'f{i}' for i in [
    298, 285, 356, 253, 171, 250, 144, 379, 51, 120, 261, 279, 53, 141, 56, 301,
    148, 104, 20, 380, 63, 202, 241, 57, 342, 223, 116, 127, 366, 178, 115, 336,
    182, 124, 9, 87, 229, 335, 338, 92, 23, 281, 69, 324, 131, 224, 201, 343, 151,
    330, 181, 344, 257, 278, 352, 358, 138, 39, 226, 291, 212, 73, 145, 353, 123,
    309, 45, 105, 373, 371, 29, 55, 350, 362, 180, 21, 205, 189, 378, 28, 31, 271,
    282, 13, 302, 337, 361, 177, 155, 364, 59, 365, 132, 35, 383, 58, 106, 34, 89,
    50, 93, 174, 52, 8, 331, 78, 72, 284, 348, 153, 230, 54, 237, 312, 300, 259,
    208, 159, 70, 220, 7, 43, 222, 292, 192, 254, 67, 340, 255, 183, 4, 207, 218,
    99, 283, 158, 187, 103, 36, 211
]]
COLS_CORR = COLS_INIT[:20]
T_ROLLING = 239  # rolling 窗口大小


# ============================================================
# 特征工程 - 与 data_processor.py 保持一致
# ============================================================

def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """添加特征：rolling 统计 + 市场平均 + timeid 别名."""
    cols = COLS_CORR
    n = T_ROLLING

    # Rolling mean 和 rolling std
    df = df.with_columns(
        [pl.col(c).rolling_mean(n).over("stockid").alias(f"{c}_rolling_avg_{n}") for c in cols] +
        [pl.col(c).rolling_std(n).over("stockid").alias(f"{c}_rolling_std_{n}") for c in cols]
    )

    # 计算 diff
    df = df.with_columns(
        [(pl.col(c) - pl.col(f"{c}_rolling_avg_{n}")).alias(f"{c}_diff_rolling_avg_{n}") for c in cols]
    )

    # 删除 rolling_avg 列
    df = df.drop([f"{c}_rolling_avg_{n}" for c in cols])

    # Cross-sectional mean
    df = df.with_columns(
        [pl.col(c).mean().over(["dateid", "timeid"]).alias(f"{c}_avg_per_date_time") for c in cols]
    )

    # TimeID 别名作为特征
    df = df.with_columns(pl.col("timeid").alias("feature_timeid"))

    return df




# ============================================================
# Dataset for DataLoader - 简化版
# ============================================================

class SimpleDataset(Dataset):
    """简化版 Dataset，按日期索引返回数据。

    返回格式：(X, y) 其中 y 包含所有标签 [LabelA, LabelB, LabelC]
    """

    def __init__(self, df: pl.DataFrame, features: list, labels: list):
        self.features = features
        self.labels = labels
        self.dateids = df['dateid'].unique().sort().to_list()

        # 按日期分组存储数据
        self._df = df.sort('stockid', 'timeid')

    def __len__(self):
        return len(self.dateids)

    def __getitem__(self, idx):
        dateid = self.dateids[idx]
        df_day = self._df.filter(pl.col('dateid') == dateid)

        # 获取维度
        n_stocks = df_day['stockid'].n_unique()
        n_times = df_day['timeid'].n_unique()

        # 提取特征和标签，reshape 为 (n_stocks, n_times, n_features)
        X = df_day.select(self.features).to_torch(dtype=pl.Float32)
        X = X.reshape(n_stocks, n_times, -1)

        y = df_day.select(self.labels).to_torch(dtype=pl.Float32)
        y = y.reshape(n_stocks, n_times, -1)

        return X, y


def collate_fn(batches: list):
    """简单的拼接函数，将多个 batch 沿 batch 维度拼接"""
    X_batch = torch.cat([b[0] for b in batches], dim=0)
    y_batch = torch.cat([b[1] for b in batches], dim=0)
    return X_batch, y_batch


# ============================================================
# 训练函数
# ============================================================

def train_and_get_score(
    model, train_dataloader, val_dataloader,
    num_features, lr=0.001, early_stopping=True, early_stopping_patience=1,
    checkpoint_interval=5, checkpoint_name="model",
    epochs=None, verbose=True
):
    """训练模型并返回验证集 score

    Args:
        num_features: 特征数量（用于日志）
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = WeightedR2Loss()
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if verbose:
        print(f"Device: {device}")
        print(f"{'Epoch':^5} | {'Train Loss':^10} | {'Val Loss':^8} | {'Train R2':^9} | {'Val R2':^7} | {'LR':^7}")
        print("-" * 60)

    min_val_r2, best_epoch, no_improvement, best_model_state = -np.inf, 0, 0, None

    # 训练循环
    for epoch in range(epochs):
        # === train_one_epoch ===
        model.train()
        total_loss = 0.0
        y_total, weights_total, preds_total = [], [], []
        itr = tqdm(train_dataloader) if verbose else train_dataloader

        for x_batch, y_batch in itr:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # 从 y_batch 拆分标签：主任务 + 辅助任务
            y_main = y_batch[:, :, 0]           # LabelA
            resp_batch = y_batch[:, :, 1:]      # LabelB, LabelC
            weights_batch = torch.ones_like(y_main).to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                out_y, out_resp, _ = model(x_batch, None, use_aux_heads=True)
                loss1 = criterion(out_y.flatten(), y_main.flatten(), weights_batch.flatten())
                if out_resp is not None:
                    loss2 = criterion(out_resp[:, :, 0].flatten(), resp_batch[:, :, 0].flatten(), weights_batch.flatten())
                    loss3 = criterion(out_resp[:, :, 1].flatten(), resp_batch[:, :, 1].flatten(), weights_batch.flatten())
                    loss = loss1 + loss2 + loss3
                else:
                    loss = loss1

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            y_total.append(y_main.flatten())
            weights_total.append(weights_batch.flatten())
            preds_total.append(out_y.detach().flatten())

        y_total = torch.cat(y_total).cpu()
        weights_total = torch.cat(weights_total).cpu()
        preds_total = torch.cat(preds_total).cpu()
        train_r2 = r2_weighted_torch(y_total, preds_total, weights_total).item()
        train_loss = total_loss / len(train_dataloader)

        # Clear training tensors
        del y_total, weights_total, preds_total

        # === validate_one_epoch ===
        model.eval()
        losses, all_y, all_weights, all_preds = [], [], [], []
        itr_val = tqdm(val_dataloader) if verbose else val_dataloader

        for x_batch, y_batch in itr_val:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_main = y_batch[:, :, 0]
            weights_batch = torch.ones_like(y_main).to(device)

            with torch.no_grad(), torch.amp.autocast('cuda'):
                preds_batch, _, _ = model(x_batch, None, use_aux_heads=False)
                loss = criterion(preds_batch.flatten(), y_main.flatten(), weights_batch.flatten())

            losses.append(loss.item())
            all_y.append(y_main.flatten())
            all_weights.append(weights_batch.flatten())
            all_preds.append(preds_batch.flatten())

        all_y = torch.cat(all_y)
        all_weights = torch.cat(all_weights)
        all_preds = torch.cat(all_preds)
        val_loss = np.mean(losses)
        val_r2 = r2_weighted_torch(all_y, all_preds, all_weights).item()

        # Clear validation tensors and switch back to train mode
        del all_y, all_weights, all_preds, losses
        model.train()

        lr_last = optimizer.param_groups[0]["lr"]
        if verbose:
            print(f"{epoch+1:^5} | {train_loss:^10.4f} | {val_loss:^8.4f} | {train_r2:^9.4f} | {val_r2:^7.4f} | {lr_last:^7.5f}")

        if val_r2 > min_val_r2:
            min_val_r2, best_model_state, no_improvement, best_epoch = val_r2, model.state_dict(), 0, epoch
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_{checkpoint_name}_epoch{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_r2': min_val_r2
                }, checkpoint_path)
                if verbose:
                    print(f"  [Checkpoint] Saved at epoch {epoch+1} (Val R2: {min_val_r2:.4f})")
        else:
            no_improvement += 1

        if early_stopping and no_improvement >= early_stopping_patience + 1:
            if verbose:
                print(f"Early stopping on epoch {best_epoch+1}. Best score: {min_val_r2:.4f}")
            break

    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return min_val_r2


# ============================================================
# 主函数
# ============================================================

def main():
    """主入口：准备数据和模型，然后训练"""
    print("=" * 60)
    print(f"训练：{MODEL_TYPE}_{HIDDEN_SIZES[0]} | 简化版训练脚本")
    print("=" * 60)

    # 配置 debug 模式
    debug = True  # 设置为 True 使用少量数据测试

    # 构建特征列表
    suffixes = [f"_diff_rolling_avg_{T_ROLLING}", f"_rolling_std_{T_ROLLING}", "_avg_per_date_time"]
    features = COLS_INIT + [f"{c}{s}" for c in COLS_CORR for s in suffixes] + ["feature_timeid"]

    # 标签列表：主任务 + 辅助任务
    labels = [COL_TARGET] + COLS_RESPONDERS  # [LabelA, LabelB, LabelC]

    # 加载数据
    print("\n加载数据...")
    if debug:
        print("[DEBUG] 使用少量数据进行测试...")

    df_lazy = pl.scan_parquet(PATH_PARQUET).sort('stockid', 'dateid', 'timeid')

    if debug:
        stockids = df_lazy.select('stockid').unique().collect().to_series().to_numpy()[:3]
        dateids = df_lazy.select('dateid').unique().sort('dateid').collect().to_series().to_numpy()[:10]
        df = df_lazy.filter(
            pl.col('stockid').is_in(stockids.tolist()) &
            pl.col('dateid').is_in(dateids.tolist())
        ).collect()
        train_days = 8  # Debug mode: 8 train + 2 validation
        print(f"[DEBUG] 只加载 3 个 stock 前 10 天 (8 train + 2 val)")
    else:
        df = pl.read_parquet(PATH_PARQUET)
        train_days = TRAIN_DAYS

    # 添加特征
    print("添加特征...")
    df = add_features(df)

    # 标准化
    print("标准化...")
    feature_cols = [col for col in df.columns if col.startswith('f')]

    # 先填充 NaN 和 null（原始数据可能有 NaN）
    df = df.with_columns([
        pl.col(col).fill_nan(0.0).fill_null(0.0) for col in feature_cols
    ])

    # 再标准化
    df = df.with_columns([
        (pl.col(col) - pl.col(col).mean()) / (pl.col(col).std() + 1e-9)
        for col in feature_cols
    ])

    # 处理 rolling 产生的 NaN/null
    df = df.with_columns([
        pl.col(col).fill_nan(0.0).fill_null(0.0) for col in feature_cols
    ])

    print(f"数据：{df.shape}, 特征：{len(features)}")

    for seed in SEED_RANGE:
        epochs = EPOCHS_PER_SEED.get(seed, 8)

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 数据划分
        df_train = df.filter(pl.col(COL_DATE) < train_days)
        df_valid = df.filter(pl.col(COL_DATE) >= train_days)

        # 创建 Dataset
        train_dataset = SimpleDataset(df_train, features, labels)
        val_dataset = SimpleDataset(df_valid, features, labels)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        print(f"\n训练集：{len(train_dataset)} 天，验证集：{len(val_dataset)} 天，特征数：{len(features)}")

        # 创建模型
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ModelR(
            len(features),
            HIDDEN_SIZES,
            DROPOUT_RATES,
            HIDDEN_SIZES_LINEAR,
            DROPOUT_RATES_LINEAR,
            MODEL_TYPE
        ).to(device)

        print(f"\n模型参数：{sum(p.numel() for p in model.parameters()):,}")

        # 训练
        score = train_and_get_score(
            model, train_dataloader, val_dataloader,
            num_features=len(features),
            lr=LR,
            early_stopping=EARLY_STOPPING,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            checkpoint_name=f"{MODEL_TYPE}_{HIDDEN_SIZES[0]}_seed{seed}",
            epochs=epochs,
            verbose=True
        )
        print(f"\nScore: {score:.5f}")

        # 清理当前 seed 的数据
        del train_dataset, val_dataset, train_dataloader, val_dataloader, model
        gc.collect()

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
