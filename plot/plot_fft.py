import gc
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

gc.collect()

LABELS = ["LabelA", "LabelB", "LabelC"]
COLORS = ["#1f77b4", "#2ca02c", "#9467bd"]

# 懒读取，只取 exchangeid=0、stockid=0 的全量数据
df = (
    pl.scan_parquet("./data/train.parquet")
    .filter((pl.col("exchangeid") == 0) & (pl.col("stockid") == 0))
    .select(["dateid", "timeid"] + LABELS)
    .sort(["dateid", "timeid"])
    .collect()
)

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for ax, label, color in zip(axes, LABELS, COLORS):
    series = df.get_column(label).drop_nulls().to_numpy().astype(np.float64)
    print(f"exchangeid=0 {label} 有效点数: {len(series)}")

    N = len(series)
    freqs = np.fft.fftshift(np.fft.fftfreq(N))
    real = np.fft.fftshift(np.fft.fft(series)).real

    ax.plot(freqs, real, linewidth=0.4, color=color)
    ax.set_ylabel("Real component")
    ax.set_title(f"FFT Real Component — {label} (exchangeid=0, all data)")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.grid(True, alpha=0.3)

    del series, real
    gc.collect()

axes[-1].set_xlabel("Normalized frequency (cycles/sample)")
plt.tight_layout()
out = "fft_labels.png"
plt.savefig(out, dpi=150)
print(f"图已保存至 {out}")

del df
gc.collect()
