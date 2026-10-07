import tempfile
from pathlib import Path
from examples.basic_race import build_config
from f1sim.ml.dataset import generate_dataset, DatasetConfig, load_dataset
from f1sim.ml.models import train_xgboost, predict_xgboost, train_torch, predict_torch

def test_dataset_and_models():
    cfg=build_config()
    with tempfile.TemporaryDirectory() as d:
        p=generate_dataset(cfg,Path(d)/'x.csv',DatasetConfig(rows=12,simulations_per_row=2,seed=1))
        df=load_dataset(p)
        assert len(df)==12
        x=train_xgboost(df.iloc[:9]); pred=predict_xgboost(x,df.iloc[9:])
        assert all(len(v)==3 for v in pred.values())
        m=train_torch(df.iloc[:9],epochs=2); tp=predict_torch(m,df.iloc[9:])
        assert all(len(v)==3 for v in tp.values())
