"""Low-latency inference backends for the trained strategy model."""
from __future__ import annotations
import time, json
from pathlib import Path
import numpy as np
from .models import encode_frame, TARGETS, predict_xgboost

def benchmark_torch(model, df, batch_sizes=(1,8,32,128,512,1024,4096), warmup=20, repeats=100):
    import torch
    X=torch.tensor(encode_frame(df).to_numpy(),dtype=torch.float32)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=model.to(device).eval(); X=X.to(device)
    rows=[]
    for b in batch_sizes:
        xb=X[:b]
        with torch.inference_mode():
            for _ in range(min(warmup,10)): model(xb)
        if device.type=='cuda': torch.cuda.synchronize()
        t=time.perf_counter()
        n=max(1,repeats if b<=128 else 20)
        with torch.inference_mode():
            for _ in range(n): model(xb)
        if device.type=='cuda': torch.cuda.synchronize()
        elapsed=(time.perf_counter()-t)/n
        rows.append({'backend':'pytorch','device':device.type,'batch_size':b,'latency_ms':elapsed*1000,'throughput':b/elapsed,'memory_mb':float(torch.cuda.memory_allocated()/2**20) if device.type=='cuda' else None})
    return rows

def try_onnx(model, df, out_path='ml/checkpoints/model.onnx'):
    try:
        import torch, onnx
    except Exception as exc: return {'available':False,'reason':str(exc)}
    X=torch.tensor(encode_frame(df).to_numpy(),dtype=torch.float32)
    torch.onnx.export(model,X, out_path, input_names=['features'],output_names=['outputs'],dynamic_axes={'features':{0:'batch'},'outputs':{0:'batch'}},opset_version=17)
    return {'available':True,'path':out_path}
