import gc
import polars as pl
import matplotlib.pyplot as plt

# 清理内存
gc.collect()

DATEID = 300
STOCKID = 0
CONFIGS = [
    ("LabelA", 10, "#1f77b4"),
    ("LabelB", 30, "#2ca02c"),
    ("LabelC", 60, "#d62728"),
]
LABELS = [c[0] for c in CONFIGS]

df = (
    pl.scan_parquet("./data/train.parquet")
    .filter((pl.col("dateid") == DATEID) & (pl.col('exchangeid')==0) & (pl.col("stockid") == STOCKID))
    .select(["timeid"] + LABELS)
    .sort("timeid")
    .collect()
)

print(f"dateid={DATEID}, stockid={STOCKID}: {len(df)} 行")

fig, ax = plt.subplots(figsize=(14, 5))

for label, shift, color in CONFIGS:
    shifted = df.select([
        pl.col("timeid"),
        pl.col(label).shift(shift).alias("val"),
    ]).drop_nulls()
    ax.plot(
        shifted["timeid"].to_numpy(),
        shifted["val"].to_numpy(),
        color=color,
        linewidth=0.8,
        label=f"{label} shift {shift}",
    )

ax.set_xlabel("timeid")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title(f"LabelA shift -10 / LabelB shift -30 / LabelC shift -60  (dateid={DATEID}, stockid={STOCKID})")
plt.tight_layout()

out = "label_shift_d0_s0.png"
plt.savefig(out, dpi=150)
print(f"图已保存至 {out}")

# 释放内存
del df
gc.collect()
