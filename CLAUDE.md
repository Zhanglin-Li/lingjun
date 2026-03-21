# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lingjun China A-Share Market Microstructure Prediction competition. 500 stocks, 384 features (`f0`-`f383`), targets `LabelA` (main), `LabelB`/`LabelC` (auxiliary).

## Documentation

- `docs/lingjun_comp.md` - Competition description and rules
- `docs/solution_lingjun.md` - Adapted solution approach for this competition
- `docs/solution_jane_street.md` - Original Jane Street solution (reference)
- `docs/exchange_analysis.md` - Analysis of exchangeid=0 vs exchangeid=1 data differences

## Environment & Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (uv recommended)
uv sync

# Run training
python run_training_simple.py     # Main training script (debug mode by default)

# Syntax check
python -m py_compile run_training_simple.py
```

## Code Architecture

### Directory Structure

```
├── custom_dataset.py       # CustomDataset (lazy loading with Polars, rolling features)
├── model.py                # ModelR, WeightedR2Loss, r2_weighted_torch (standalone)
├── run_training_simple.py  # Main training script (hardcoded config)
├── pyproject.toml
└── data/                   # train.parquet, test.parquet
```

### Key Components

**Data Flow:**

1. Load `train.parquet` → Add rolling features → Standardize → Fill NaN
2. `SimpleDataset` (in run_training_simple.py) - Wraps DataFrame, groups by dateid
3. Features: 140 selected features + rolling stats + cross-sectional means

**Model Architecture (`model.py`):**

- `ModelRBase`: GRU/LSTM layers + FC head
- `ModelR`: Dual-GRU with auxiliary heads for multi-task learning
- `WeightedR2Loss`: Loss function for training

**Training (`run_training_simple.py`):**

- Hardcoded config at top of file (no YAML)
- Debug mode: 3 stocks × 10 days (8 train + 2 val)
- Early stopping patience=1, checkpoint every 5 epochs

### Configuration

All config hardcoded in `run_training_simple.py`:
- `PATH_PARQUET = "./data/train.parquet"`
- `COL_TARGET = "LabelA"`, `COLS_RESPONDERS = ["LabelB", "LabelC"]`
- `HIDDEN_SIZES = [500]`, `LR = 0.0005`, `BATCH_SIZE = 16`
- `TRAIN_DAYS = 288`, `VALID_DAYS = 72`
- `debug = True` for quick testing

### Data Specifications

- Input: `train.parquet` (~43M rows)
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
r2 = 1 - sum(w * (pred - true) ** 2) / (sum(w * true ** 2) + 1e-38)
```