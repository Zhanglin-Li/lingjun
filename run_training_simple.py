#!/usr/bin/env python3
"""简化版训练脚本 - 只依赖基础包 (polars/numpy/tqdm/torch)

配置从 config.yaml 读取。
"""
import csv
import gc
import os
import random
import copy
from datetime import datetime

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm.auto import tqdm

from model import ModelR, WeightedR2Loss, r2_weighted_torch


def set_seed(seed: int):
    """固定所有随机性来源，确保可复现"""
    # Python 内置 random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    """DataLoader worker 进程的随机种子初始化"""
    worker_info = get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed  # PyTorch 自动传递的种子
        np.random.seed(seed)
        random.seed(seed)


# ============================================================
# 实验日志记录
# ============================================================

class ExperimentLogger:
    """实验日志记录器，记录超参和训练结果到 CSV 文件

    CSV 文件同时作为 checkpoint 代号的映射表。
    checkpoint 命名格式: exp_{id:03d}.pt
    """

    LOG_FILE = "experiments/experiment_log.csv"
    FIELDS = [
        "exp_id", "timestamp", "model_type", "hidden_sizes", "dropout_rates",
        "hidden_sizes_linear", "dropout_rates_linear", "lr", "batch_size",
        "weight_decay", "early_stopping_patience", "num_features",
        "best_epoch", "best_val_r2", "final_train_loss", "final_val_loss",
        "total_epochs", "ckpt_path", "notes"
    ]

    def __init__(self):
        os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)
        if not os.path.exists(self.LOG_FILE):
            with open(self.LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.FIELDS)

    def get_next_id(self) -> int:
        """获取下一个实验 ID"""
        if not os.path.exists(self.LOG_FILE):
            return 1
        with open(self.LOG_FILE, "r") as f:
            lines = f.readlines()
            return len(lines)  # 行数 = id (header 不算)

    def log(self, exp_id: int, config: dict, results: dict, ckpt_path: str = "", notes: str = ""):
        """记录一次实验

        Args:
            exp_id: 实验代号
            config: 包含超参的字典
            results: 包含训练结果的字典
            ckpt_path: checkpoint 文件路径
            notes: 备注信息
        """
        row = {
            "exp_id": exp_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": config.get("model_type", ""),
            "hidden_sizes": str(config.get("hidden_sizes", [])),
            "dropout_rates": str(config.get("dropout_rates", [])),
            "hidden_sizes_linear": str(config.get("hidden_sizes_linear", [])),
            "dropout_rates_linear": str(config.get("dropout_rates_linear", [])),
            "lr": config.get("lr", 0),
            "batch_size": config.get("batch_size", 0),
            "weight_decay": config.get("weight_decay", 0.01),
            "early_stopping_patience": config.get("early_stopping_patience", 0),
            "num_features": config.get("num_features", 0),
            "best_epoch": results.get("best_epoch", 0),
            "best_val_r2": f"{results.get('best_val_r2', 0):.6f}",
            "final_train_loss": f"{results.get('final_train_loss', 0):.4f}",
            "final_val_loss": f"{results.get('final_val_loss', 0):.4f}",
            "total_epochs": results.get("total_epochs", 0),
            "ckpt_path": ckpt_path,
            "notes": notes,
        }

        with open(self.LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)

        print(f"📊 实验已记录: exp_{exp_id:03d}")


# ============================================================
# 配置部分（从 config.yaml 读取）
# ============================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """从 YAML 文件加载配置"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# 支持通过环境变量指定配置文件路径
import sys
_config_path = os.environ.get("CONFIG_PATH", "config.yaml")
if len(sys.argv) > 1:
    _config_path = sys.argv[1]

# 加载配置
_config = load_config(_config_path)
print(f"📄 使用配置文件: {_config_path}")

# 数据路径
PATH_PARQUET = _config["data"]["path_parquet"]

# 列名配置
COL_TARGET = _config["data"]["col_target"]
COL_ID = _config["data"]["col_id"]
COL_DATE = _config["data"]["col_date"]
COL_TIME = _config["data"]["col_time"]
COL_WEIGHT = _config["data"]["col_weight"]
COLS_RESPONDERS = _config["data"]["cols_responders"]

# 模型配置
MODEL_TYPE = _config["model"]["type"]
HIDDEN_SIZES = _config["model"]["hidden_sizes"]
DROPOUT_RATES = _config["model"]["dropout_rates"]
HIDDEN_SIZES_LINEAR = _config["model"]["hidden_sizes_linear"]
DROPOUT_RATES_LINEAR = _config["model"]["dropout_rates_linear"]

# 训练配置
BATCH_SIZE = _config["training"]["batch_size"]
LR = _config["training"]["lr"]
LR_ONLINE = _config["training"]["lr_online"]
EPOCHS_PER_SEED = {int(k): v for k, v in _config["training"]["epochs_per_seed"].items()}
SEED_RANGE = range(max(EPOCHS_PER_SEED.keys()) + 1)
EARLY_STOPPING = _config["training"]["early_stopping"]
EARLY_STOPPING_PATIENCE = _config["training"]["early_stopping_patience"]
CHECKPOINT_INTERVAL = _config["training"]["checkpoint_interval"]
ONLINE_LEARNING = _config["training"]["online_learning"]

# 数据划分
TRAIN_DAYS = _config["split"]["train_days"]
VALID_DAYS = _config["split"]["valid_days"]
TOTAL_DAYS = TRAIN_DAYS + VALID_DAYS

# 特征配置
COLS_INIT = _config["features"]["cols_init"]
COLS_CORR = COLS_INIT[:len(COLS_INIT)//5]  # 1/5 features for rolling/cross-sectional stats
T_ROLLING = _config["features"]["t_rolling"]


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


class GPUDataset(Dataset):
    """GPU 常驻 Dataset - 所有数据预加载到 GPU，消除传输瓶颈。

    适用场景：GPU 内存足够容纳全部数据时使用。
    数据必须预先按 (dateid, stockid, timeid) 排序。
    """

    def __init__(self, df: pl.DataFrame, features: list, labels: list, device: torch.device):
        self.features = features
        self.labels = labels
        self.device = device

        # 获取日期列表
        dateids = df['dateid'].unique().sort().to_list()
        self.dateids = dateids
        n_days = len(dateids)

        # 预处理所有数据到 GPU
        print(f"预加载 {n_days} 天数据到 GPU...")

        # 先在 GPU 分配空间
        n_stocks = 500  # 固定 500 stocks
        n_times = 239   # 固定 239 times
        self.X_gpu = torch.empty(n_days, n_stocks, n_times, len(features), device=device)
        self.y_gpu = torch.empty(n_days, n_stocks, n_times, len(labels), device=device)

        # 数据已排序，直接分批处理
        chunk_size = 10  # 每次处理 10 天
        for start in range(0, n_days, chunk_size):
            end = min(start + chunk_size, n_days)
            date_chunk = dateids[start:end]

            # 用 range filter（数据已排序）
            df_chunk = df.filter(
                (pl.col('dateid') >= date_chunk[0]) & (pl.col('dateid') <= date_chunk[-1])
            )
            X_chunk = df_chunk.select(features).to_torch(dtype=pl.Float32)
            X_chunk = X_chunk.reshape(len(date_chunk), n_stocks, n_times, -1)

            y_chunk = df_chunk.select(labels).to_torch(dtype=pl.Float32)
            y_chunk = y_chunk.reshape(len(date_chunk), n_stocks, n_times, -1)

            # 移动到 GPU 并同步
            self.X_gpu[start:end] = X_chunk.to(device)
            self.y_gpu[start:end] = y_chunk.to(device)
            torch.cuda.synchronize()

            # 清理 CPU 内存
            del X_chunk, y_chunk, df_chunk
            import gc
            gc.collect()

            print(f"  已加载 {end}/{n_days} 天...")

        # 记录实际形状
        self.n_stocks = n_stocks
        self.n_times = n_times

        print(f"GPU 数据加载完成: X {self.X_gpu.shape}, y {self.y_gpu.shape}")

    def __len__(self):
        return len(self.dateids)

    def __getitem__(self, idx):
        # 直接从 GPU tensor 切片，无传输开销
        return self.X_gpu[idx], self.y_gpu[idx]


def collate_fn(batches: list):
    """简单的拼接函数，将多个 batch 沿 batch 维度拼接"""
    X_batch = torch.cat([b[0] for b in batches], dim=0)
    y_batch = torch.cat([b[1] for b in batches], dim=0)
    return X_batch, y_batch


# ============================================================
# 训练函数
# ============================================================

def run_epoch(model, dataloader, criterion, device, optimizer=None,
              scaler=None, use_aux_heads=True, online_lr=None, verbose=False,
              data_on_gpu=False):
    """Run one epoch.

    Args:
        optimizer: If provided, runs training mode with backward pass
        use_aux_heads: Use auxiliary heads for multi-task learning
        online_lr: If provided, do online learning update (single backward, no optimizer)
        data_on_gpu: If True, data is already on GPU (skip .to(device))

    Returns:
        tuple: (avg_loss, r2_score)
    """
    is_training = optimizer is not None or online_lr is not None
    model.train() if is_training else model.eval()

    total_loss, n_batches = 0.0, 0
    ss_res = 0.0  # Σ(y_pred - y_true)²
    all_y = []  # 收集所有 y_true 用于计算均值
    iterator = tqdm(dataloader) if verbose else dataloader

    for x_batch, y_batch in iterator:
        # 数据已在 GPU 时跳过传输
        if not data_on_gpu:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

        y_main = y_batch[:, :, 0]
        weights = torch.ones_like(y_main)

        context = torch.no_grad() if not is_training else torch.amp.autocast('cuda')

        with context:
            preds, out_resp, _ = model(x_batch, None, use_aux_heads=use_aux_heads)

            # Compute loss
            loss = criterion(preds.flatten(), y_main.flatten(), weights.flatten())
            if use_aux_heads and out_resp is not None:
                resp = y_batch[:, :, 1:]
                loss += criterion(out_resp[:, :, 0].flatten(), resp[:, :, 0].flatten(), weights.flatten())
                loss += criterion(out_resp[:, :, 1].flatten(), resp[:, :, 1].flatten(), weights.flatten())

        # Training update
        if optimizer is not None:
            optimizer.zero_grad()
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

        # Online learning update (validation with AdamW, LabelA only)
        # Follows JS approach: single forward/backward with AdamW + grad clipping
        elif online_lr is not None:
            online_optimizer = torch.optim.AdamW(model.parameters(), lr=online_lr, weight_decay=0.01)
            online_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            online_optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # 收集预测和真实值用于 R² 计算
        with torch.no_grad():
            ss_res += ((preds - y_main) ** 2).sum().item()
            all_y.append(y_main.detach().cpu())

    avg_loss = total_loss / n_batches

    # 计算标准 R²: 1 - Σ(y_pred - y_true)² / Σ(y_true - y_mean)²
    all_y_cat = torch.cat(all_y)
    y_mean = all_y_cat.mean().item()
    ss_tot = ((all_y_cat - y_mean) ** 2).sum().item()
    r2 = 1 - ss_res / (ss_tot + 1e-38)

    return avg_loss, r2


def train_and_get_score(
    model, train_dataloader, val_dataloader,
    lr=0.001, early_stopping=True, early_stopping_patience=1,
    exp_id=0,
    epochs=None, verbose=True, online_learning=False, data_on_gpu=False
):
    """训练模型并返回验证集 score.

    Args:
        online_learning: If True, update model during validation with LR_ONLINE
        data_on_gpu: If True, data is already on GPU (skip .to(device))
        exp_id: 实验代号，用于 checkpoint 命名

    Returns:
        dict: 包含 best_val_r2, best_epoch, total_epochs, final_train_loss, final_val_loss,
              ckpt_path (checkpoint 路径)
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = WeightedR2Loss()
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if verbose:
        print(f"Device: {device}")
        header = f"{'Epoch':^5} | {'Train Loss':^10} | {'Val Loss':^8} | {'Train R2':^9} | {'Val R2':^7}"
        if online_learning:
            header += f" | {'Online':^7}"
        print(header)
        print("-" * (67 if online_learning else 57))

    min_val_r2, best_epoch, no_improvement, best_model_state = -np.inf, 0, 0, None
    ckpt_path = None
    online_lr = LR_ONLINE if online_learning else None

    # 用于记录最终 epoch 的指标
    final_train_loss, final_val_loss = 0.0, 0.0

    for epoch in range(epochs):
        # Train epoch
        train_loss, train_r2 = run_epoch(
            model, train_dataloader, criterion, device,
            optimizer=optimizer, scaler=scaler, use_aux_heads=True,
            verbose=verbose, data_on_gpu=data_on_gpu
        )

        # Validation epoch (with optional online learning on a COPY of the model)
        # Follow Jane Street approach: deepcopy model before online learning updates
        # to prevent validation updates from affecting subsequent training
        if online_lr is not None:
            model_val = copy.deepcopy(model)
            val_loss, val_r2 = run_epoch(
                model_val, val_dataloader, criterion, device,
                use_aux_heads=False, online_lr=online_lr,
                verbose=verbose, data_on_gpu=data_on_gpu
            )
        else:
            val_loss, val_r2 = run_epoch(
                model, val_dataloader, criterion, device,
                use_aux_heads=False,
                verbose=verbose, data_on_gpu=data_on_gpu
            )

        # 更新最终 epoch 的指标
        final_train_loss, final_val_loss = train_loss, val_loss

        if verbose:
            msg = f"{epoch+1:^5} | {train_loss:^10.4f} | {val_loss:^8.4f} | {train_r2:^9.4f} | {val_r2:^7.4f}"
            if online_learning:
                msg += f" | {'✓':^7}"
            print(msg)

        # Early stopping & checkpointing
        if val_r2 > min_val_r2:
            # 删除旧的 best checkpoint
            if ckpt_path is not None and os.path.exists(ckpt_path):
                os.remove(ckpt_path)
                if verbose:
                    print(f"  [Clean] Removed old checkpoint: {ckpt_path}")

            min_val_r2, best_model_state, no_improvement, best_epoch = val_r2, model.state_dict(), 0, epoch

            # 保存最佳模型，使用 exp_id 命名
            ckpt_path = f"{checkpoint_dir}/exp_{exp_id:03d}.pt"
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': min_val_r2,
            }, ckpt_path)
            if verbose:
                print(f"  [Best] Saved: {ckpt_path} (R²={min_val_r2:.6f})")
        else:
            no_improvement += 1

        if early_stopping and no_improvement >= early_stopping_patience + 1:
            if verbose:
                print(f"Early stopping at epoch {best_epoch+1}. Best: {min_val_r2:.4f}")
            break

    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "best_val_r2": min_val_r2,
        "best_epoch": best_epoch + 1,
        "total_epochs": epoch + 1,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "ckpt_path": ckpt_path,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    """主入口：准备数据和模型，然后训练"""
    print("=" * 60)
    print(f"训练：{MODEL_TYPE}_{HIDDEN_SIZES[0]} | 简化版训练脚本")
    print("=" * 60)

    # 配置 debug 模式
    debug = False  # 设置为 False 使用完整数据

    # 构建特征列表
    suffixes = [f"_diff_rolling_avg_{T_ROLLING}", f"_rolling_std_{T_ROLLING}", "_avg_per_date_time"]
    features = COLS_INIT + [f"{c}{s}" for c in COLS_CORR for s in suffixes] + ["feature_timeid"]

    # 标签列表：主任务 + 辅助任务
    labels = [COL_TARGET] + COLS_RESPONDERS  # [LabelA, LabelB, LabelC]

    # 缓存文件路径（根据 debug 模式区分）
    cache_dir = "./data/cache"
    cache_file = f"{cache_dir}/train_standardized{'_debug' if debug else ''}.parquet"
    os.makedirs(cache_dir, exist_ok=True)

    # 检查是否存在缓存文件
    if os.path.exists(cache_file):
        print(f"\n发现缓存文件: {cache_file}")
        print("直接读取已标准化的数据...")
        df = pl.read_parquet(cache_file)
        train_days = TRAIN_DAYS
        print(f"数据：{df.shape}")
    else:
        # 加载数据
        print("\n加载数据...")
        if debug:
            print("[DEBUG] 使用少量数据进行测试...")

        # 需要读取的列：特征列 + 元数据列 + 标签列
        meta_cols = [COL_ID, COL_DATE, COL_TIME]
        label_cols = [COL_TARGET] + COLS_RESPONDERS
        select_cols = meta_cols + label_cols + COLS_INIT

        df_lazy = pl.scan_parquet(PATH_PARQUET).select(select_cols).sort('stockid', 'dateid', 'timeid')

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
            df = df_lazy.collect()
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

        # 保存缓存
        print(f"保存缓存到: {cache_file}")
        df.write_parquet(cache_file)
        print(f"数据：{df.shape}, 特征：{len(features)}")

    for seed in SEED_RANGE:
        epochs = EPOCHS_PER_SEED.get(seed, 8)

        # 固定所有随机性来源
        set_seed(seed)

        # 为 DataLoader 创建独立的随机生成器
        g = torch.Generator()
        g.manual_seed(seed)

        # 创建设备（GPUDataset 需要）
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 预先排序（避免在 GPUDataset 中重复排序）
        print("排序数据...")
        df_sorted = df.sort('dateid', 'stockid', 'timeid')

        # 数据划分（已排序）
        df_train = df_sorted.filter(pl.col(COL_DATE) < train_days)
        df_valid = df_sorted.filter(pl.col(COL_DATE) >= train_days)

        # 创建 Dataset（使用 GPU 预加载版）
        train_dataset = GPUDataset(df_train, features, labels, device)
        val_dataset = GPUDataset(df_valid, features, labels, device)

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn,  # 不需要 pin_memory，数据已在 GPU
            generator=g, worker_init_fn=worker_init_fn
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn
        )

        print(f"\n训练集：{len(train_dataset)} 天，验证集：{len(val_dataset)} 天，特征数：{len(features)}")

        # 创建模型
        model = ModelR(
            len(features),
            HIDDEN_SIZES,
            DROPOUT_RATES,
            HIDDEN_SIZES_LINEAR,
            DROPOUT_RATES_LINEAR,
            MODEL_TYPE
        ).to(device)

        print(f"\n模型参数：{sum(p.numel() for p in model.parameters()):,}")

        # 获取实验代号
        logger = ExperimentLogger()
        exp_id = logger.get_next_id()
        print(f"实验代号: exp_{exp_id:03d}")

        # 训练
        results = train_and_get_score(
            model, train_dataloader, val_dataloader,
            lr=LR,
            early_stopping=EARLY_STOPPING,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            exp_id=exp_id,
            epochs=epochs,
            verbose=True,
            online_learning=ONLINE_LEARNING,
            data_on_gpu=True  # 数据已在 GPU
        )

        # 记录实验
        config = {
            "model_type": MODEL_TYPE,
            "hidden_sizes": HIDDEN_SIZES,
            "dropout_rates": DROPOUT_RATES,
            "hidden_sizes_linear": HIDDEN_SIZES_LINEAR,
            "dropout_rates_linear": DROPOUT_RATES_LINEAR,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "weight_decay": 0.01,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "num_features": len(features),
        }
        logger.log(exp_id, config, results, ckpt_path=results["ckpt_path"], notes=f"seed{seed}")

        # 清理当前 seed 的数据
        del train_dataset, val_dataset, train_dataloader, val_dataloader, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
