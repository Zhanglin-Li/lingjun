import polars as pl
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

df = (
    pl.scan_parquet("./data/train.parquet")
    .filter(pl.col("stockid") == 0)
    .select(["dateid", "timeid", "LabelA", "LabelB", "LabelC"])
    .sort(["dateid", "timeid"])
    .collect()
)

print(f"stockid=0 共 {len(df)} 行")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for ax, label in zip(axes, ["LabelA", "LabelB", "LabelC"]):
    series = df[label].drop_nulls().to_numpy()
    plot_acf(series, lags=120, ax=ax, title=f"ACF - {label} (stockid=0)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")

plt.tight_layout()
plt.savefig("acf_stockid0.png", dpi=150)
print("图已保存至 acf_stockid0.png")
# plt.show()
