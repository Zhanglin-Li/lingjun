# Hidden Size Sweep Plan

## Context
已知两层 GRU 强于单层（exp_011: R2=0.0102, exp_012: R2=0.0102 vs exp_009: R2=0.003）。
但不确定 2 层时最优 hidden size，以及增大层数是否更好。

## Settings
- 所有层 dropout = 0（简化调参）
- LR = 0.0001, BS = 4, features = 70 (top 10% importance)
- patience = 10, epochs = 200, online_learning = true
- split: train 288 days, val 72 days

## Sweep Configs

| Config | Layers | Hidden Sizes | Linear Dropout |
|--------|--------|--------------|----------------|
| s01 (baseline) | 2 | [512, 256] | [0.0, 0.0] |
| s02 | 2 | [256, 128] | [0.0, 0.0] |
| s03 | 2 | [384, 192] | [0.0, 0.0] |
| s04 | 2 | [768, 384] | [0.0, 0.0] |
| s05 | 2 | [1024, 512] | [0.0, 0.0] |
| s06 | 3 | [384, 192, 96] | [0.0, 0.0] |
| s07 | 3 | [512, 256, 128] | [0.0, 0.0] |
| s08 | 3 | [768, 384, 192] | [0.0, 0.0] |
| s09 | 4 | [512, 256, 128, 64] | [0.0, 0.0] |
| s10 | 4 | [384, 192, 96, 48] | [0.0, 0.0] |

## Running
```bash
# Run all configs sequentially with resume support
bash scripts/run_sweep.sh

# If interrupted, just re-run — already-completed configs are skipped
bash scripts/run_sweep.sh
```

## Results
| Config | Val R2 | Notes |
|--------|--------|-------|
| s01 | TBD | baseline re-run (compare with exp_012) |
