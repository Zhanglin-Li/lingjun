# CLAUDE.md

## Configuration Modification Rule (CRITICAL)

**配置文件参数（如 batch_size、lr、hidden_sizes 等）不能随意修改。**
- 只有用户明确要求"改配置"时才能修改
- 修改前必须先询问用户确认
- 代码工程优化（如内存优化、并行处理）不需要询问

## Project Overview

Lingjun China A-Share Market Microstructure Prediction competition. 500 stocks, 384 features (`f0`-`f383`), targets `LabelA` (main), `LabelB`/`LabelC` (auxiliary).

**Competition Notes:**
- Data partitioned by `stockid` (500 stocks)
- Submission format: `stockid|dateid|timeid,prediction`
- Last 10 timeids (229-238) not scored but must be submitted
- No forward-looking: only use current/earlier dateid+timeid data

## Documentation

详细项目文档（架构、数据流、特征工程、模型、训练、Online Learning 等）见 `docs/README.md`。

### Directory Layout

- **`config/`** — baseline configs (template.yaml, config.yaml)
- **`config/<param>/`** — sweep configs per parameter (e.g. config/hidden_size/)
- **`scripts/`** — all shell scripts (e.g. scripts/run_sweep.sh)

## Environment & Commands

```bash
# Activate & install
source .venv/bin/activate && uv sync

# Run training
.venv/bin/python train.py config/template.yaml    # default, takes config as first arg
# Also: train.ipynb (HuggingFace Trainer, Jupyter notebook)
# Also: bash scripts/run_sweep.sh (runs config/*.yaml sequentially with resume support)

# Sweep configs: config/<param>/ directory per sweep (e.g. config/hidden_size/)
# Scripts: scripts/ directory (e.g. scripts/run_sweep.sh)

# Syntax check
.venv/bin/python -m py_compile train.py

# Clear cache (when switching datasets)
rm -f ./data/cache/train_standardized.parquet
```

## Package Installation (CRITICAL)

**必须使用 uv，禁止 pip install！禁止使用国外官网源！**

```bash
# 安装新包（默认使用阿里云镜像，已在 pyproject.toml 配置）
source .venv/bin/activate && uv add <package>

# 国内镜像源（默认生效，无需额外指定）
# 阿里云: http://mirrors.aliyun.com/pypi/simple（pyproject.toml 已配置为默认）
# 清华: https://pypi.tuna.tsinghua.edu.cn/simple

# 仅作为备选：uv pip install（用于版本冲突时）
source .venv/bin/activate && uv pip install <package>
```

**禁止使用：**
- `pip install` - 会绕过 uv lock，导致依赖混乱
- 国外官网源（如 https://download.pytorch.org/whl）- 国内无法连接或极慢

## Training Patterns

- **Primary training**: `train.py` (HuggingFace Trainer, Python script) — takes config path as first arg
- **Notebook**: `train.ipynb` — same HF Trainer setup, uses `config/template.yaml`
- **Sweep automation**: `scripts/run_sweep.sh` runs configs sequentially; checks `experiments/experiment_log.csv` line count to skip completed runs; interrupted sweeps resume by re-running
- **Config format**: OmegaConf, needs `data.n_features`, `model.{hidden_sizes,dropout_rates,hidden_sizes_linear,dropout_rates_linear,type}`, `training.{per_device_train_batch_size,early_stopping_patience,num_train_epochs,lr}`
- **GPU memory limit**: 24GB GPU — hidden sizes [768, 384] fit, [1024, 512] OOM; 3+ layers with [512, 256, 128] fit, [768, 384, 192] OOM
- **Best known config**: 2-layer [768, 384], dropout=0, LR=1e-4, BS=4 → Val R2 ≈ -0.0103
- **Long epoch pattern**: `Dataset.__len__` returns `actual_dates * repeat`; train uses `repeat=1000`, val uses `repeat=1`
- **Dataset split**: train dates 0-299, val dates 300-359
- **Validation**: R2 score computed at each eval step on val set
- **Steps-based eval/save**: `eval_strategy='steps'`, `eval_steps=100`, `save_strategy='steps'`, `save_steps=100`
- **Early stopping**: HuggingFace `EarlyStoppingCallback` with `patience` on val R2 (`metric_for_best_model='eval_r2'`, `load_best_model_at_end=True`)
- **HuggingFace constraint**: `load_best_model_at_end=True` requires `save_strategy == eval_strategy`
- **Fast experiment mode**: `early_stopping_patience=1` for quick LR/parameter sweeps
- **Dataset**: `LingjunTrainSet` loads parquet with cross-sectional zscore normalization; `__getitem__` returns `data[index % len]` to cycle dates; `repeat` param controls dataset length multiplier

### ModelR `use_aux_targets` (CRITICAL)

- **`ModelR` has `use_aux_targets=False` by default**: No auxiliary heads; loss is only LabelA R2
- **When `True`**: Creates aux heads for LabelB/LabelC, adds their R2 losses to total loss
- **All model instantiation sites** pass `getattr(cfg, 'use_aux_targets', False)` — default OFF

### HF Trainer Integration (CRITICAL)

- **Model forward must return dict**: `return {"loss": loss, "logits": main_pred}` — returning a tuple `(loss,)` causes HF Trainer to extract empty logits via `outputs[1:]`, skipping `compute_metrics`
- **`compute_metrics` shapes**: `predictions` is `(N, T)`, `labels` is `(N, T, 3)` for [LabelA, LabelB, LabelC]. Must extract LabelA first:
  ```python
  predictions = predictions.reshape(-1)
  labels = labels[:, :, 0].reshape(-1)  # LabelA
  return {'r2': r2_score(labels, predictions)}
  ```

## Jupyter MCP (CRITICAL)

**Jupyter server 的 `base_url` 是 `/jupyter/`，MCP 默认连接 `http://localhost:8888` 会 404。**
每次新会话需要使用 Jupyter MCP 时，必须先执行以下连接命令：

```python
mcp__jupyter__connect_to_jupyter(
    jupyter_url="http://localhost:8888/jupyter",
    jupyter_token="jupyter-autodl-container-ab4341a2e0-0e2de444-e6581ff7706eb44528af95f4101762a354bf3c84b6feb43fd942bb72f29536103"
)
```

**禁止使用 `http://localhost:8888`（缺少 `/jupyter/` 前缀），否则会返回 404 错误。**
**注意 URL 末尾不要带斜杠**，否则会出现 `//` 双斜杠错误。

已安装 `jupyter-collaboration`（v4.3.0），使 MCP 的 `execute_cell`/`read_cell` 等 notebook 操作正常工作。

可用工具：`connect_to_jupyter`, `use_notebook`, `execute_code`, `execute_cell`, `insert_cell`, `edit_cell_source`, `read_cell`, `list_notebooks`, `read_notebook`, `list_files`。

## Bash Command Style

- Put comments AFTER the command, not inside quoted strings
- Avoid `#` inside `-c "..."` Python strings - explain in response text
- **禁止使用 `tail -N`、`head -N`、`grep --color` 等截断或过滤命令的输出** — 用户需要看到命令的完整输出信息，不要自行截断
- 运行 Python 脚本优先用 `.venv/bin/python script.py`，而非 `source .venv/bin/activate && python script.py`（后者在后台任务中容易丢失激活状态）