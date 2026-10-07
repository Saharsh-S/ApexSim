"""Benchmark ML inference backends; unavailable optional runtimes are reported, never fabricated."""
from __future__ import annotations
import argparse, json, time, os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC=os.path.join(ROOT,"src")
if SRC not in sys.path: sys.path.insert(0,SRC)
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from pathlib import Path
import numpy as np
from examples.basic_race import build_config
from f1sim.ml.dataset import generate_dataset, DatasetConfig, load_dataset
from f1sim.ml.models import train_xgboost, train_torch, predict_xgboost
from f1sim.ml.inference import benchmark_torch, try_onnx

def main():
 p=argparse.ArgumentParser(); p.add_argument('--rows',type=int,default=4096); p.add_argument('--sims',type=int,default=10); p.add_argument('--out',default='benchmarks/inference/phase8_results.json'); a=p.parse_args()
 cfg=build_config(); path='ml/datasets/phase8_bench.csv'; generate_dataset(cfg,path,DatasetConfig(a.rows,a.sims,19)); df=load_dataset(path)
 split=max(1,int(len(df)*.8)); train=df.iloc[:split]; test=df.iloc[split:]
 xgb=train_xgboost(train)
 results=[]
 for b in (1,8,32,128,512,1024,4096):
  xb=test.iloc[:min(b,len(test))]
  n=1000 if b<=128 else 100
  t=time.perf_counter()
  for _ in range(n): predict_xgboost(xgb,xb)
  sec=(time.perf_counter()-t)/n
  results.append({'backend':'xgboost_cpu','batch_size':b,'latency_ms':sec*1000,'throughput':len(xb)/sec,'memory_mb':None})
 torch_model=train_torch(train,epochs=20); results.extend(benchmark_torch(torch_model,test))
 # Precision experiments are measured on the active PyTorch device only.
 import torch
 for dtype_name, dtype in [('fp16', torch.float16), ('bf16', torch.bfloat16)]:
  try:
   pm=torch_model.to(dtype).eval(); X=torch.tensor(__import__('f1sim.ml.models',fromlist=['encode_frame']).encode_frame(test).to_numpy(),dtype=dtype)
   if next(pm.parameters()).is_cuda: X=X.cuda()
   for b in (1,8,32,128,512,1024,4096):
    xb=X[:min(b,len(X))]; n=200 if b<=128 else 30
    with torch.inference_mode():
     for _ in range(10): pm(xb)
    if xb.is_cuda: torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.inference_mode():
     for _ in range(n): pm(xb)
    if xb.is_cuda: torch.cuda.synchronize()
    sec=(time.perf_counter()-t)/n
    results.append({'backend':'pytorch_'+dtype_name,'batch_size':b,'latency_ms':sec*1000,'throughput':len(xb)/sec,'memory_mb':float(torch.cuda.memory_allocated()/2**20) if xb.is_cuda else None})
  except Exception as exc:
   results.append({'backend':'pytorch_'+dtype_name,'available':False,'reason':str(exc)})
 torch_model=torch_model.float()
 results.append({'backend':'onnx','status':try_onnx(torch_model,test.iloc[:1])})
 try:
  import tensorrt
  results.append({'backend':'tensorrt','available':True})
 except Exception as e: results.append({'backend':'tensorrt','available':False,'reason':str(e)})
 try:
  import triton
  results.append({'backend':'triton','available':True,'note':'Optional Triton kernel module is present; dense MLP fusion is only enabled on a supported CUDA device.'})
 except Exception as e: results.append({'backend':'triton','available':False,'reason':str(e)})
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(results,indent=2)); print(out)
if __name__=='__main__': main()
