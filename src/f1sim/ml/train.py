"""Train and save Phase 7 models."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
from .dataset import generate_dataset, DatasetConfig, load_dataset
from .models import train_xgboost, train_torch

def main():
 p=argparse.ArgumentParser(); p.add_argument('--rows',type=int,default=1000); p.add_argument('--sims',type=int,default=20); p.add_argument('--out',default='ml/datasets/strategy_outcomes.csv'); a=p.parse_args()
 from examples.basic_race import build_config
 cfg=build_config(); path=generate_dataset(cfg,a.out,DatasetConfig(a.rows,a.sims))
 df=load_dataset(path); split=int(len(df)*.8); train=df.iloc[:split].copy(); test=df.iloc[split:].copy()
 models=train_xgboost(train); Path('ml/checkpoints').mkdir(parents=True,exist_ok=True); joblib.dump(models,'ml/checkpoints/xgboost.joblib')
 torch_model=train_torch(train); import torch; torch.save(torch_model.state_dict(),'ml/checkpoints/torch_model.pt')
 print(json.dumps({'rows':len(df),'train':len(train),'test':len(test),'checkpoint':'ml/checkpoints'},indent=2))
if __name__=='__main__': main()
