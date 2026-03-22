# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lingjun China A-Share Market Microstructure Prediction competition. 500 stocks, 384 features (`f0`-`f383`), targets `LabelA` (main), `LabelB`/`LabelC` (auxiliary).

## Documentation

- `docs/lingjun_comp.md` - Competition description and rules
- `docs/solution_lingjun.md` - Adapted solution approach for this competition
- `docs/solution_jane_street.md` - Original Jane Street solution (reference)
- `docs/exchange_analysis.md` - Analysis of exchangeid=0 vs exchangeid=1 data differences
- `kagglejanestreet/` - JS original source code (reference for online learning, model architecture)

## Environment & Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (uv recommended)
uv sync

# Run training
python run_training_simple.py config/config.yaml  # Pass config as argument
# Or via environment: CONFIG_PATH=config/config.yaml python run_training_simple.py

# Syntax check
python -m py_compile run_training_simple.py
```

## Code Architecture

### Directory Structure

```
├── config/                  # Configuration files
│   ├── config.yaml          # Default config (best: exp_012)
│   ├── config_exp_014.yaml  # GRU [512, 256] 2层递减
│   ├── config_exp_015.yaml  # GRU [512, 512] 2层非递减
│   ├── config_exp_016.yaml  # GRU [512, 256, 128] 3层递减
│   ├── config_exp_017.yaml  # GRU [512, 512, 512] 3层非递减
│   ├── config_exp_018.yaml  # GRU [768, 384, 192] 3层增大+递减
│   └── config_exp_019.yaml  # LSTM [512, 256]
├── custom_dataset.py        # CustomDataset (lazy loading with Polars)
├── model.py                 # ModelR, WeightedR2Loss, r2_weighted_torch
├── run_training_simple.py   # Main training script (YAML config)
├── pyproject.toml
├── data/                    # train.parquet, test.parquet
├── experiments/             # experiment_log.csv
└── checkpoints/             # exp_XXX.pt
```

### Key Components

**Data Flow:**

1. Load `train.parquet` → Add rolling features → Standardize → Fill NaN
2. `GPUDataset` (in run_training_simple.py) - Preloads data to GPU, wraps DataFrame by dateid
3. Features: 140 selected features + rolling stats + cross-sectional means

**Model Architecture (`model.py`):**

- `ModelRBase`: GRU/LSTM layers + FC head
- `ModelR`: Dual-GRU with auxiliary heads for multi-task learning
- `WeightedR2Loss`: Loss function for training

**Training (`run_training_simple.py`):**

- YAML config file (pass as argument: `python run_training_simple.py config/config_exp_014.yaml`)
- exp_id auto-increments based on experiment_log.csv
- Checkpoint naming: `exp_{id:03d}.pt`
- Online learning: AdamW (lr=0.0003, weight_decay=0.01) + grad clipping (max_norm=1.0) per batch

### Configuration

Config files in `config/` folder. Run with:
```bash
python run_training_simple.py config/config_exp_014.yaml

# When switching datasets (e.g., filtered.parquet → train.parquet), delete cache:
rm -f ./data/cache/train_standardized.parquet
```

Default settings:
- `path_parquet = "data/train.parquet"` (full dataset, ~43M rows)
- `COL_TARGET = "LabelA"`, `COLS_RESPONDERS = ["LabelB", "LabelC"]`
- `lr = 0.0001`, `lr_online = 0.00006`, `batch_size = 4`
- `online_learning = true`
- `early_stopping_patience = 10`

### Data Specifications

- Input: `train.parquet` (~43M rows, 42GB)
- GPU Memory for GPUDataset:
  - 224 stocks (filtered.parquet): BS=4 safe on 32GB VRAM
  - 500 stocks (train.parquet): BS=1 required, 22.7GB total data
- Pre-sort data before GPUDataset to avoid Polars sort memory spike
- Time steps: T=239 per sample
- Train/Val split: 288/72 days
- Rolling window: 239 time steps

### Exchange Analysis (IMPORTANT)

**exchangeid=0 vs exchangeid=1 are TOTALLY DIFFERENT:**
- **Different stocks**: 224 vs 276, zero overlap (each stock belongs to exactly ONE exchange)
- **Different row counts**: 19.2M vs 23.7M
- **ALL 384 features have DIFFERENT means** between exchanges
- **NaN patterns differ**: Some features NaN in one exchange but not the other (e.g., f200 NaN only in exch 0)
- **Stock IDs are interleaved** between exchanges (not sequential ranges)

**Implications:**
- Filter by `exchangeid=0` for primary training (or model exchanges separately)
- Exchange-specific normalization may improve performance
- Do NOT assume feature distributions are identical across exchanges
- When analyzing data, use explicit per-exchange filtering (not `group_by().agg()`)

### Competition Notes

- Data partitioned by `stockid` (500 stocks: interleaved across exchanges)
- Each stock belongs to exactly ONE exchange (no overlap)
- Submission format: `stockid|dateid|timeid,prediction`
- Last 10 timeids (229-238) not scored but must be submitted
- No forward-looking: only use current/earlier dateid+timeid data

## Common Patterns

**NaN handling (CRITICAL):**

```python
# train.parquet has NaN in some features (e.g., f285)
# Must use fill_nan() - fill_null() does NOT handle NaN
df = df.with_columns([
    pl.col(col).fill_nan(0.0).fill_null(0.0) for col in feature_cols
])
# After standardization, fill again (rolling stats produce NaN)
df = df.with_columns([
    pl.col(col).fill_nan(0.0).fill_null(0.0) for col in feature_cols
])
```

**Feature engineering:**

```python
# Rolling statistics per stockid
pl.col(col).rolling_mean(239).over('stockid')
pl.col(col).rolling_std(239).over('stockid')
# Cross-sectional mean per datetime
pl.col(col).mean().over('dateid', 'timeid')
```

**Tensor reshaping for RNN:**

```python
# Shape: (n_stocks, n_times, n_features) → (batch, time, feature)
X = X.reshape(T, -1, K).swapaxes(0, 1)  # For batch-first RNN
```

**Weighted R²:**

```python
# 标准 R² 公式: R² = 1 - SS_res / SS_tot
y_mean = sum(w * y_true) / sum(w)
ss_res = sum(w * (pred - true) ** 2)
ss_tot = sum(w * (true - y_mean) ** 2)
r2 = 1 - ss_res / (ss_tot + 1e-38)
```

## Bash Command Style

- Put comments/explanations AFTER the command, not inside quoted strings
- Avoid `#` inside `-c "..."` Python strings - explain in response text instead
- This prevents permission approval prompts

---

## Hyperparameter Tuning Progress

Experiments logged in `experiments/experiment_log.csv`. Checkpoint naming: `exp_{id:03d}.pt`.

### Completed Experiments (exp_001 - exp_013)

Using `data/filtered.parquet` (exchangeid=0 subset):

| ID | hidden_sizes | dropout | LR | BS | Best Val R² | Epoch | Notes |
|----|--------------|---------|-----|-----|-------------|-------|-------|
| 1 | [500] | 0.3 | 0.0005 | 16 | -0.0167 | 10 | Baseline |
| 4 | [500] | 0.3 | 0.0001 | 16 | +0.0047 | 37 | First positive R² |
| 8 | [500] | 0.3 | 0.0001 | 4 | +0.0032 | 36 | BS=4 baseline |
| 9 | [256] | 0.3 | 0.0001 | 4 | +0.0030 | 19 | Smaller capacity |
| 10 | [768] | 0.3 | 0.0001 | 4 | -0.0087 | 9 | Larger → overfit |
| 11 | [512, 256] | 0.3 | 0.0001 | 4 | **+0.0102** | 57 | **Best!** |
| 12 | [512, 256] | 0.1 | 0.0001 | 4 | **+0.0102** | 37 | Same good |

### Planned Experiments (exp_014 - exp_019)

Using `data/train.parquet` (full dataset, 500 stocks), with online learning enabled:

| ID | Config File | Model | hidden_sizes | Description |
|----|-------------|-------|--------------|-------------|
| 14 | config_exp_014.yaml | GRU | [512, 256] | patience=15 (filtered) |
| 15 | config_exp_014.yaml | GRU | [512, 256] | full dataset, BS=1 (in progress) |
| 16 | config_exp_016.yaml | GRU | [512, 256, 128] | 3层递减 |
| 17 | config_exp_017.yaml | GRU | [512, 512, 512] | 3层非递减 |
| 18 | config_exp_018.yaml | GRU | [768, 384, 192] | 3层增大+递减 |
| 19 | config_exp_019.yaml | LSTM | [512, 256] | 对比GRU |

Run command:
```bash
python run_training_simple.py config/config_exp_014.yaml
```

### Key Findings

1. **Feature count**: 80 features (129 total with rolling stats) is optimal. More features = more noise.

2. **Learning rate**: **lr=0.0001** is optimal. Higher LR (0.0003-0.002) all negative R².

3. **Model capacity**: **Two-layer GRU [512, 256]** outperforms single-layer.
   - Single [768] overfits quickly, single [256] underfits
   - Two-layer allows deeper representation without overfitting

4. **Dropout**: **0.1-0.3** works well, **0.5 too aggressive**.

5. **Online learning**: AdamW (lr=0.0003, weight_decay=0.01) + grad clipping (max_norm=1.0) per batch.

### Current Best Configuration

```yaml
model:
  type: gru
  hidden_sizes: [512, 256]
  dropout_rates: [0.1, 0.1, 0.0]
  hidden_sizes_linear: [500, 300]
  dropout_rates_linear: [0.2, 0.1]

training:
  batch_size: 4
  lr: 0.0001
  lr_online: 0.00006
  online_learning: true
  early_stopping_patience: 10

features:
  cols_init: Top 80 features by importance
```