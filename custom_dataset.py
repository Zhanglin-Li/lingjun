import torch
from torch.utils.data import DataLoader, Dataset
import polars as pl
from typing import Optional, Sequence
import polars.selectors as cs 

class CustomDataset(Dataset):
    def __init__(
        self,
        path: str,
        window: int = 239,
        drop_cols: Sequence[str] = ("exchangeid",),
        debug: bool = False,
        start_dateid: int = 0,
        end_dateid: Optional[int] = None,
        rolling: Optional[list] = None,
        standardize: bool = True
      
    ):

        self.window = window
        self.drop_cols = drop_cols
        self.lazy_df = pl.scan_parquet(path).sort('stockid','dateid','timeid').drop(drop_cols)
        self.start_dateid, self.end_dateid = start_dateid,end_dateid
        self.rolling = rolling
        exprs = []
        if self.debug:
            self.lazy_df = self.lazy_df.filter([
                pl.col('stockid')<=3 ,
                pl.col('dateid')<=3
            ])
        if self.rolling is not None:
            for col in self.rolling:
                exprs.extend(
                    [
                        pl.col(col).rolling_mean(self.window).over('stockid').alias(f'{col}_rm'),
                        pl.col(col).rolling_std(self.window).over('stockid').alias(f'{col}_rstd'),
                        pl.col(col).mean().over('dateid', 'timeid').alias(f'{col}_crm'),
                    ]
                )
            self.lazy_df = self.lazy_df.with_columns(exprs)

        if standardize:
            self.lazy_df = self.lazy_df.with_columns([
                (pl.col(col)-pl.col(col).mean())/(pl.col(col).std()+1e-9) 
                for col in self.lazy_df.collect_schema().to_python() if col.startswith('f')])

        self.lazy_df=self.lazy_df.collect()
     
    def __len__(self) -> int:
        if self.end_dateid is None:
            self.end_dateid = self.lazy_df.select(pl.col('dateid').max()).collect().to_numpy()[0,0]
        return self.end_dateid-self.start_dateid+1

    def __getitem__(self, dateid: int):
       
        df = self.lazy_df.filter(pl.col('dateid')==dateid+self.start_dateid)
        df = df.with_columns(cs.starts_with('f').fill_null(0).fill_nan(0))
        # df = df.collect()

        n_stockid = df['stockid'].n_unique()
        n_timeid = df['timeid'].n_unique()
        X = df.select(pl.col('timeid'),cs.starts_with('f')).to_torch(dtype = pl.Float32)
        X = X.reshape(n_stockid,n_timeid,-1)
        y = df.select(cs.starts_with('Label')).to_torch(dtype = pl.Float32)
        y = y.reshape(n_stockid,n_timeid,-1)
        return X, y


def collate_fn(batches:list):
    return (torch.cat([b[0] for b in batches]),torch.cat([b[1] for b in batches]))
if __name__ == "__main__":
    data_path = "./data/filtered.parquet"
    # 360 days, 8:2 train_val_split, [start_dateid,end_dateid] closed interval
    drop_cols = pl.read_csv('./feature_importance/feature_importance_10pct.csv').filter(pl.col('rank')>=80)['feature'].to_list()
    rolling = pl.read_csv('./feature_importance/feature_importance_10pct.csv').filter(pl.col('rank')<=15)['feature'].to_list()
    debug = False
    
    train_ds = CustomDataset(
        path=data_path,
        start_dateid=0,
        end_dateid=359-72-1,
        window=239,
        debug=debug,
        drop_cols = drop_cols,
        rolling=rolling
    )
    # val_ds = CustomDataset(
    #     path=data_path,
    #     start_dateid=359-72,
    #     end_dateid=359,
    #     window=239,
    #     debug=debug,
    #     drop_cols=drop_cols,
        # rolling=rolling
    # )
    train_loader = DataLoader(train_ds,batch_size=1,shuffle=True,pin_memory=True,collate_fn=collate_fn)
    # val_loader = DataLoader(val_ds,batch_size=1,shuffle=False,pin_memory=True,collate_fn=collate_fn,num_workers=workers)

    # print("dataset init ok")
    # print("len(ds) =", len(train_ds))
    # print("T =", train_ds.T)

    for i,b in enumerate(train_loader):
        print(f'order {i}:')
        print(b[0].shape)
        print(b[1].shape)
        if i>=10: break