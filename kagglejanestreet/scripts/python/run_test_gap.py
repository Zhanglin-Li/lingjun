"""Load model and test it on a sample of last 200 days with a gap of 200 days.
"""
import numpy as np
import polars as pl
from tqdm.auto import tqdm

from janestreet.setup_env import setup_environment
from janestreet.pipeline import FullPipeline
from janestreet.models.nn import NN
from janestreet.config import COL_DATE, COL_WEIGHT, COL_TARGET
from janestreet.data_processor import DataProcessor
from janestreet.metrics import r2_weighted
from janestreet.transformers import PolarsTransformer


MODEL_TYPE = "gru"
NUM = 3

setup_environment()

data_processor = DataProcessor(f"{MODEL_TYPE}_{NUM}.{0}_{700}_cv", skip_days=1200)
df = data_processor.get_train_data()

features = data_processor.features

print(features)
print(f"Number of features: {len(features)}")

preds = []
for NUM in [2, 3]:
    for SEED in range(3):
        MODEL_NAME = f"{MODEL_TYPE}_{NUM}.{SEED}_{700}_cv"
        print(MODEL_NAME)

        model = NN(random_seed=SEED)

        pipeline = FullPipeline(
            model,
            preprocessor=PolarsTransformer(features),
            run_name="fold0",
            name=MODEL_NAME,
            load_model=True,
            features=features,
            refit=True,
            change_lr=False,
        )
        df_test = df.filter((pl.col(COL_DATE) >= 1499)&(pl.col(COL_DATE) < 1699))
        df_valid = df.filter(pl.col(COL_DATE) >= 1299)

        pipeline.fit(verbose=True)

        cnt_dates = 0
        preds_m = []
        dates = np.unique(df_valid.select(pl.col(COL_DATE)).to_series().to_numpy())
        for date_id in tqdm(dates):
            df_valid_date = df.filter(pl.col(COL_DATE) == date_id)
            if pipeline.refit & (cnt_dates > 0):
                df_valid_time = df.filter(pl.col(COL_DATE) == date_id-1)
                pipeline.update(df_valid_time)
            if date_id >= 1499:
                preds_i, hidden = pipeline.predict(df_valid_date, n_times=None)
                preds_m += list(preds_i)
            cnt_dates += 1

        preds_m = np.array(preds_m)
        preds.append(preds_m)

preds = np.mean(preds, axis=0)
y = df_test.select(pl.col(COL_TARGET)).to_series().to_numpy()
weight = df_test.select(pl.col(COL_WEIGHT)).to_series().to_numpy()
score = r2_weighted(y, preds, weight)
print(f"Score: {score:.5f}")
