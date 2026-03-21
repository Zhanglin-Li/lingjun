# Solution Plan for Lingjun China A-Share Market Prediction

This document describes the methodology adapted from the Jane Street solution for the Lingjun competition.

## 1. Cross-validation

Use time-series CV with two folds:
- Fold 0: Earlier dates for training, later dates for validation
- Fold 1: Validate on the last dates with a gap to simulate the test dataset scenario
- Validation size: ~72 dates (matching the train/val split)

**Lingjun-specific note:** The competition has 239 time steps per day (`timeid` 0-238), and predictions for the last 10 time steps (229-238) are not evaluated, but must still be submitted.

## 2. Feature Engineering and Data Preparation

### 2.1 Sample

- Data is partitioned by `stockid` (500 stocks, 0-499)
- All stocks can be used for training
- `exchangeid` is a categorical feature indicating the stock's exchange
- Filter by `exchangeid=0` for primary training data (using `filtered.parquet` or `filter.parquet`)

### 2.2 Data Preparation

Apply simple standardization and NaN imputation with zero. For the 384 microstructure features (`f0` - `f383`), normalization is important due to the heavy-tailed distributions typical of high-frequency data.

### 2.3 Feature Engineering

Use all original features (`f0` - `f383`) plus the following engineered features:

**Market averages:**
```python
# Cross-sectional mean per datetime
pl.col(feature).mean().over('dateid', 'timeid')
```
Captures market-wide movements and regime shifts.

**Rolling statistics:**
```python
# Rolling statistics per stock (window = 239, one full day)
pl.col(feature).rolling_mean(239).over('stockid')
pl.col(feature).rolling_std(239).over('stockid')
```
The window size of 239 matches the number of intraday time steps (adapted from Jane Street's 1000).

**Time feature:**
- Add `timeid` as a feature to capture intraday patterns

**Categorical handling:**
- `exchangeid` can be used as a categorical feature or to filter data

## 3. Model Architecture

### 3.1 Base Model

Time-series GRU with sequence equal to one day (239 time steps). Train two architectures:

- **Model 1:** 3-layer GRU
- **Model 2:** 1-layer GRU followed by 2 linear layers with ReLU activation and dropout

Both models contribute to the final ensemble.

### 3.2 Auxiliary Labels

Use `LabelB` and `LabelC` as auxiliary targets for multi-task learning:

```python
COL_TARGET = "LabelA"
COLS_RESPONDERS = ["LabelB", "LabelC"]
```

**Training approach:**
- Use a separate base model for each auxiliary target
- Pass predictions through a linear layer to produce the final target output (`LabelA`)
- Sum of losses (weighted R²) for each label:
  - Main loss: R² on `LabelA`
  - Auxiliary losses: R² on `LabelB` and `LabelC`

**Training parameters:**
- Batch size: one day (group by `dateid`)
- Learning rate: 0.0005
- Early stopping with patience=1

For submission, train models on data up to the last `dateid`, using the number of epochs equal to the average optimal number of epochs on CV.

### 3.3 Ensemble

Run both model architectures on 3 different seeds and take a simple unweighted average of predictions from those 6 models.

## 4. Online Learning

**Note:** The Lingjun competition requires source code submission and reproducibility. Online learning applicability depends on the inference setup.

If online learning is allowed during inference:
- When new data with targets becomes available, perform one forward pass to update model weights
- Learning rate for updates: 0.0003
- Updates are performed only with the `LabelA` loss (no auxiliary targets)

## 5. Technical Details

### 5.1 Inference Speed

Key optimizations:
- Use PyTorch with optimized data loading
- Cache rolling statistics to avoid recomputation
- Use Polars for efficient data processing

### 5.2 Technical Stack

- **Data processing:** Polars (lazy evaluation, memory efficient)
- **Deep learning:** PyTorch
- **Experiment tracking:** WandB
- **Compute:** vast.ai or similar GPU instances

### 5.3 Data Loading

```python
# Data is partitioned by stockid in parquet format
import polars as pl

df = pl.scan_parquet("data/train.parquet")
# Filter by exchangeid if needed
df = df.filter(pl.col("exchangeid") == 0)
```

## 6. Submission Format

The submission file format:

```
stockid|dateid|timeid,prediction
0|0|0,0.0
0|0|1,0.0
...
```

**Important:**
1. Predictions for `timeid` 229-238 are NOT evaluated, but must still be submitted
2. No forward-looking: only use data at current `dateid` and `timeid` or earlier
3. Source code must be provided for reproducibility

## 7. Key Differences from Jane Street Competition

| Aspect | Jane Street | Lingjun |
|--------|-------------|---------|
| Target | `responder_6` | `LabelA` |
| Auxiliary labels | `responder_7`, `responder_8` + derived | `LabelB`, `LabelC` |
| Features | ~80 features + categorical | 384 features (`f0`-`f383`) |
| Entity ID | `symbol_id` | `stockid` (0-499) |
| Time steps/day | ~968 | 239 (`timeid` 0-238) |
| Rolling window | 1000 time_ids | 239 time_ids (one day) |
| Metric | Weighted zero-mean R² | R-Squared |
| Data partition | By symbol | By stockid |
| Exchange filter | N/A | `exchangeid=0` filter available |
| Online learning | Yes | TBD (code submission required) |

## 8. Scores

| Model | CV fold 0 | CV fold 1 | Fold 1 with gap | CV avg |
|-------|-----------|-----------|-----------------|--------|
| GRU baseline | - | - | - | - |
| GRU + aux labels | - | - | - | - |
| GRU ensemble (3 seeds) | - | - | - | - |

## 9. Implementation Notes

### Data Shape
- Input: `(n_stocks, 239, n_features)` - reshaped for batch-first RNN
- Each sample represents one stock over one day

### Tensor Reshaping for RNN
```python
# Shape: (n_stocks, n_times, n_features) -> (batch, time, feature)
X = X.reshape(T, -1, K).swapaxes(0, 1)
```

### Weighted R² Loss
```python
def r2_weighted(y_true, y_pred, weights):
    return 1 - (weights * (y_pred - y_true)**2).sum() / (weights * y_true**2).sum() + 1e-38
```