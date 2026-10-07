"""Small baseline and neural strategy outcome models."""
from __future__ import annotations
import numpy as np

CATEGORICAL = ["tire_compound","weather","start_tire","second_tire"]
TARGETS = ["expected_race_time","finishing_position","podium_probability","win_probability"]

def encode_frame(df):
    x=df.copy()
    for c in CATEGORICAL: x[c]=x[c].astype("category").cat.codes
    return x.drop(columns=TARGETS, errors="ignore").astype("float32")

def train_xgboost(train_df):
    from xgboost import XGBRegressor
    X=encode_frame(train_df); models={}
    for target in TARGETS:
        models[target]=XGBRegressor(n_estimators=120,max_depth=4,learning_rate=.05,subsample=.9,colsample_bytree=.9,objective="reg:squarederror",n_jobs=1,random_state=7)
        models[target].fit(X,train_df[target])
    return models

def predict_xgboost(models, df):
    X=encode_frame(df); return {t:m.predict(X) for t,m in models.items()}

def build_torch_model(input_dim, hidden=64):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,4))

def train_torch(train_df, epochs=30):
    import torch
    X=encode_frame(train_df).to_numpy(); y=train_df[TARGETS].to_numpy(dtype=np.float32)
    xt=torch.tensor(X); yt=torch.tensor(y)
    model=build_torch_model(X.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss=nn_loss=torch.nn.SmoothL1Loss()
    for _ in range(epochs):
        opt.zero_grad(); pred=model(xt); l=loss(pred,yt); l.backward(); opt.step()
    return model

def predict_torch(model, df):
    import torch
    with torch.inference_mode(): out=model(torch.tensor(encode_frame(df).to_numpy(),dtype=torch.float32)).numpy()
    return {t:out[:,i] for i,t in enumerate(TARGETS)}
