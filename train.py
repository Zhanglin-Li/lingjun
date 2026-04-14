# %%
import sys
import os
import csv
from datetime import datetime

import polars as pl
from torch.utils.data import Dataset
from typing import Optional
import torch
import yaml
from model import ModelR
from transformers import Trainer, TrainingArguments
from omegaconf import OmegaConf
from sklearn.metrics import r2_score
from transformers import EarlyStoppingCallback

# %%
config_path = sys.argv[1] if len(sys.argv) > 1 else './config/template.yaml'
print(f"Config: {config_path}")
cfg = OmegaConf.load(config_path)
# Load features and data


# %%
features = pl.read_csv('./data/feature_importance_10pct.csv')['feature'].head(cfg.data.n_features).to_list()
df = pl.read_parquet(f'./data/zscore_{cfg.data.n_features}.parquet')
print(f"Features: {len(features)}, Data shape: {df.shape}")
class LingjunTrainSet(Dataset):
    IDS = ['stockid', 'dateid', 'exchangeid', 'timeid']
    LABELS = ['LabelA', 'LabelB', 'LabelC']
    T = 239

    def __init__(self, features, start_date: int, end_date: int,
                 cached: Optional[pl.DataFrame] = None, df_path: Optional[str] = None, debug=False, repeat=1000):

        self.features = features + ['f_timeid']
        self.start_date = start_date
        self.end_date = end_date

        if cached is not None:
            self.df = cached.filter(pl.col('dateid').is_between(start_date, end_date)).with_columns(
                pl.col('timeid').alias('f_timeid'))
            self.len = len(self.df['dateid'].unique())

        else:
            assert df_path is not None
            df = pl.scan_parquet(df_path)
            df = df.with_columns(pl.col('timeid').alias('f_timeid'))

            df = df.select(self.IDS + self.LABELS + self.features).filter(
                pl.col('dateid').is_between(start_date, end_date))
            self.len = len(self.df['dateid'].unique())

            if debug:
                df = df.filter(pl.col('stockid') < 3)

            df = df.with_columns([
                (pl.col(c) - pl.col(c).mean().over('dateid', 'timeid')) / (pl.col(c).std().over('dateid', 'timeid') + 1e-9)
                for c in self.features if c != 'f_timeid'
            ])
            df = df.with_columns([
                pl.col(c).fill_nan(0).fill_null(0) for c in self.features
            ])
            self.df = df.collect()
        self.repeat = repeat
        self.data = [self.get_X_y(i) for i in range(self.len)]

    def __len__(self):
        return self.len * self.repeat

    def __getitem__(self, index):
        return self.data[index % self.len]

    def get_X_y(self, index):
        data = self.df.filter(pl.col('dateid') == (index + self.start_date))
        X = data.select(self.features).to_torch(dtype=pl.Float32).reshape(-1, self.T, len(self.features))
        y = data.select(self.LABELS).to_torch(dtype=pl.Float32).reshape(-1, self.T, len(self.LABELS))
        return {'x': X, 'labels': y}
# dateid: 0-359
train_ds = LingjunTrainSet(features=features, start_date=0, end_date=299, cached=df)
val_ds = LingjunTrainSet(features=features, start_date=300, end_date=359, cached=df, repeat=1)
print(f"Train dates: {train_ds.len}, Val dates: {val_ds.len}")
print(f"Train len: {len(train_ds)}, Val len (1x): {len(val_ds)}")


# %%
def collate_fn(pairs):
    return {'x': torch.concat([p['x'] for p in pairs]), 'labels': torch.concat([p['labels'] for p in pairs])}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions: (N, T), labels: (N, T, 3) for [LabelA, LabelB, LabelC]
    # Flatten first two dims and extract LabelA
    predictions = predictions.reshape(-1)
    labels = labels[:, :, 0].reshape(-1)  # LabelA
    return {'r2': r2_score(labels, predictions)}

# Model
model_cfg = cfg.model
model = ModelR(
    len(train_ds.features),
    model_cfg.hidden_sizes,
    model_cfg.dropout_rates,
    model_cfg.hidden_sizes_linear,
    model_cfg.dropout_rates_linear,
    model_cfg.type,
    getattr(model_cfg, 'use_aux_targets', False)
).cuda()

# Training args
tr_cfg = cfg.training
      
training_args = TrainingArguments(
    num_train_epochs=tr_cfg.num_train_epochs,
    per_device_train_batch_size=tr_cfg.per_device_train_batch_size,
    per_device_eval_batch_size=tr_cfg.per_device_eval_batch_size,
    learning_rate=tr_cfg.lr,
    lr_scheduler_type="reduce_lr_on_plateau",
    lr_scheduler_kwargs={
          "mode": "min",
          "factor": 0.5,
          "patience": 5,
          "threshold": 1e-4,
          "min_lr": 1e-7,
    },             
    eval_strategy='steps',
    eval_steps=100,
    save_strategy='no',
    logging_steps=100,
    logging_dir='./runs',
    report_to='tensorboard',
    metric_for_best_model='eval_r2',
    greater_is_better=True,
    fp16=True,
    disable_tqdm=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=tr_cfg.early_stopping_patience)]
)

# %%
trainer.train()

best_r2 = trainer.state.best_metric
print(f"\n=== RESULT ===")
print(f"lr={tr_cfg.lr}, best_model_val_r2={best_r2:.6f}")
print(f"==============\n")

# Log to experiment CSV
LOG_FILE = "experiments/experiment_log.csv"
LOG_FIELDS = [
    "exp_id", "timestamp", "config", "hidden_sizes", "dropout_rates",
    "hidden_sizes_linear", "dropout_rates_linear", "lr", "batch_size",
    "early_stopping_patience", "n_features", "num_epochs",
    "best_val_r2", "notes"
]

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_FIELDS)

# Get next exp_id
with open(LOG_FILE, "r") as f:
    exp_id = len(f.readlines())  # header counts as 1, first data row = 1

config_name = os.path.splitext(os.path.basename(config_path))[0]
dropout_rates = list(model_cfg.dropout_rates)
dropout_rates_linear = list(model_cfg.dropout_rates_linear)

row = {
    "exp_id": exp_id,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config": config_name,
    "hidden_sizes": str(list(model_cfg.hidden_sizes)),
    "dropout_rates": str(dropout_rates),
    "hidden_sizes_linear": str(list(model_cfg.hidden_sizes_linear)),
    "dropout_rates_linear": str(dropout_rates_linear),
    "lr": tr_cfg.lr,
    "batch_size": tr_cfg.per_device_train_batch_size,
    "early_stopping_patience": tr_cfg.early_stopping_patience,
    "n_features": len(train_ds.features),
    "num_epochs": tr_cfg.num_train_epochs,
    "best_val_r2": f"{best_r2:.6f}",
    "notes": config_name,
}
with open(LOG_FILE, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
    writer.writerow(row)
print(f"Logged to {LOG_FILE} as exp_{exp_id:03d}")


