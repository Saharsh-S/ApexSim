"""Controlled, reproducible Phase 10 ML Systems performance study."""
from __future__ import annotations

import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path: sys.path.insert(0, str(ROOT / "src"))

from examples.basic_race import build_config
from f1sim import Strategy, TireCompound
from f1sim.monte_carlo import simulate_strategies_numpy, simulate_strategies_python
from f1sim.gpu import cuda_available, cuda_backend_status, simulate_strategy_gpu


def _system():
    info = {"python": sys.version.split()[0], "platform": sys.platform}
    try:
        import psutil
        info.update({"cpu_count": psutil.cpu_count(logical=True), "cpu_model": None})
        try:
            info["cpu_model"] = next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")), None)
        except Exception: pass
    except Exception:
        info["cpu_count"] = os.cpu_count()
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc: info["torch_error"] = str(exc)
    info["cuda_backend_status"] = cuda_backend_status()
    info["nvidia_smi"] = shutil.which("nvidia-smi") is not None
    return info


def _cpu_util_sample(fn):
    try:
        import psutil
        p = psutil.Process()
        before = p.cpu_times()
        start = time.perf_counter(); value = fn(); elapsed = time.perf_counter() - start
        after = p.cpu_times()
        cpu_seconds = (after.user + after.system) - (before.user + before.system)
        return value, elapsed, 100.0 * cpu_seconds / elapsed / max(psutil.cpu_count(logical=True), 1)
    except Exception:
        start=time.perf_counter(); value=fn(); return value, time.perf_counter()-start, None


def _workload():
    cfg = build_config()
    strategies = [
        Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD]),
        Strategy(TireCompound.SOFT, [10], [TireCompound.MEDIUM]),
        Strategy(TireCompound.SOFT, [10, 22], [TireCompound.MEDIUM, TireCompound.HARD]),
        Strategy(TireCompound.MEDIUM, [12, 23], [TireCompound.HARD, TireCompound.MEDIUM]),
    ]
    return cfg, strategies


def benchmark_simulation(races_per_strategy=250):
    cfg, strategies = _workload()
    rows=[]
    for name, fn in (("python_cpu", simulate_strategies_python), ("numpy_cpu", simulate_strategies_numpy)):
        total=len(strategies)*races_per_strategy
        _, elapsed, cpu_util = _cpu_util_sample(lambda: fn(cfg, strategies, races_per_strategy, seed=42))
        rows.append({"implementation":name,"status":"measured","simulations":total,"runtime_s":elapsed,"throughput_sims_s":total/elapsed,"latency_ms_per_sim":elapsed/total*1000,"cpu_util_percent":cpu_util,"gpu_util_percent":None,"gpu_memory_mb":None,"energy_j":None})
    gpu_status=cuda_backend_status()
    if cuda_available():
        total=len(strategies)*races_per_strategy
        t0=time.perf_counter()
        for i,s in enumerate(strategies):
            simulate_strategy_gpu(cfg,s,races_per_strategy,seed=42+i*races_per_strategy)
        elapsed=time.perf_counter()-t0
        rows.append({"implementation":"cuda_optimized","status":"measured","simulations":total,"runtime_s":elapsed,"throughput_sims_s":total/elapsed,"latency_ms_per_sim":elapsed/total*1000,"cpu_util_percent":None,"gpu_util_percent":None,"gpu_memory_mb":None,"energy_j":None})
    else:
        rows.append({"implementation":"cuda_optimized","status":"not_run","reason":gpu_status})
    return rows


def benchmark_inference():
    rows=[]
    try:
        import torch
        from f1sim.ml.models import build_torch_model
        input_dim=15
        model=build_torch_model(input_dim).eval()
        checkpoint = ROOT / 'ml' / 'checkpoints' / 'phase8_torch.pt'
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=model.to(device)
        rng=np.random.default_rng(11)
        for precision, dtype in (("fp32",torch.float32),("fp16",torch.float16),("bf16",torch.bfloat16)):
            if precision == "bf16" and device.type == "cpu" and not hasattr(torch, 'bfloat16'):
                continue
            try: m=model.to(dtype=dtype); x=torch.tensor(rng.normal(size=(1,input_dim)).astype(np.float32),device=device,dtype=dtype)
            except Exception as exc:
                rows.append({"implementation":f"pytorch_{precision}","status":"not_run","reason":str(exc)}); continue
            for _ in range(30): m(x)
            if device.type=='cuda': torch.cuda.synchronize()
            n=500
            t0=time.perf_counter()
            for _ in range(n): m(x)
            if device.type=='cuda': torch.cuda.synchronize()
            elapsed=(time.perf_counter()-t0)/n
            mem=float(torch.cuda.max_memory_allocated()/2**20) if device.type=='cuda' else None
            rows.append({"implementation":f"pytorch_{precision}","status":"measured","batch_size":1,"runtime_s":elapsed,"latency_ms":elapsed*1000,"throughput_samples_s":1/elapsed,"device":device.type,"gpu_memory_mb":mem,"gpu_util_percent":None,"cpu_util_percent":None,"energy_j":None})
        # Batching is an explicit optimization axis; measure FP32 across fixed sizes.
        xbase=torch.randn(4096,input_dim,device=device,dtype=torch.float32); m=model.float().eval()
        for b in (1,8,32,128,512,1024,4096):
            xb=xbase[:b]
            for _ in range(20): m(xb)
            if device.type=='cuda': torch.cuda.synchronize()
            n=100 if b<=128 else 20
            t0=time.perf_counter()
            for _ in range(n): m(xb)
            if device.type=='cuda': torch.cuda.synchronize()
            elapsed=(time.perf_counter()-t0)/n
            rows.append({"implementation":"pytorch_batch_fp32","status":"measured","batch_size":b,"runtime_s":elapsed,"latency_ms":elapsed*1000,"throughput_samples_s":b/elapsed,"device":device.type,"gpu_memory_mb":float(torch.cuda.max_memory_allocated()/2**20) if device.type=='cuda' else None,"gpu_util_percent":None,"cpu_util_percent":None,"energy_j":None})
    except Exception as exc:
        rows.append({"implementation":"pytorch","status":"not_run","reason":str(exc)})
    return rows


def historical_ablation():
    p=ROOT/"benchmarks"/"results"/"phase4_cudasim_profile.json"
    if not p.exists(): return []
    d=json.loads(p.read_text())
    out=[]
    opt=d.get("deterministic_noise_optimization",{})
    if opt:
        out.append({"optimization":"conditional stochastic work","baseline_s":opt.get("before_kernel_seconds"),"optimized_s":opt.get("after_kernel_seconds"),"improvement_percent":opt.get("improvement_percent"),"environment":"Numba CUDA simulator on CPU","hardware_gpu":False})
    transfer=d.get("persistent_context_transfer_optimization",{})
    if transfer:
        out.append({"optimization":"persistent device configuration","baseline_bytes":transfer.get("before_host_to_device_bytes"),"optimized_bytes":transfer.get("after_host_to_device_bytes"),"bytes_reduction_percent":transfer.get("bytes_reduction_percent"),"component_time_reduction_percent":transfer.get("component_time_reduction_percent"),"environment":"Numba CUDA simulator on CPU","hardware_gpu":False})
    return out


def benchmark_realtime(repeats=20):
    try:
        from f1sim.realtime import RaceState, RealtimeStrategyEngine
        from f1sim.config import WeatherType
        cfg=build_config(); state=RaceState(37,4,TireCompound.MEDIUM,18,2.4,1.8,.62,.15,.42,WeatherType.DRY)
        engine=RealtimeStrategyEngine(cfg,seed=42)
        decisions=[]
        for _ in range(repeats):
            decisions.append(engine.decide(state,candidate_count=16,top_k=3,backend='simulation',simulations_per_candidate=4))
        timing={k:float(mean(getattr(d.timing,k) for d in decisions)) for k in ('state_processing_ms','candidate_generation_ms','evaluation_ms','ranking_ms','total_ms')}
        return {"status":"measured","repeats":repeats,"candidate_count":16,"simulations_per_candidate":4,"mean_stage_latency_ms":timing,"target_total_ms":100.0,"target_met":timing['total_ms'] < 100.0}
    except Exception as exc:
        return {"status":"not_run","reason":str(exc)}


def benchmark_accuracy():
    """Evaluate the persisted Phase 8 models on a freshly generated fixed-seed set."""
    out=[]
    try:
        from f1sim.ml.dataset import generate_dataset, DatasetConfig, load_dataset
        from f1sim.ml.models import build_torch_model, predict_torch, predict_xgboost, TARGETS
        cfg=build_config()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"eval.csv"
            generate_dataset(cfg,path,DatasetConfig(rows=128,simulations_per_row=8,seed=123))
            df=load_dataset(path)
            torch_model=build_torch_model(15)
            torch_model.load_state_dict(__import__('torch').load(ROOT/'ml/checkpoints/phase8_torch.pt',map_location='cpu',weights_only=True)); torch_model.eval()
            tp=predict_torch(torch_model,df)
            for target in TARGETS:
                y=df[target].to_numpy(); e=np.abs(tp[target]-y)
                out.append({"implementation":"pytorch_model","target":target,"mae":float(e.mean()),"rmse":float(np.sqrt(np.mean((tp[target]-y)**2))),"status":"measured"})
            import joblib
            xp=predict_xgboost(joblib.load(ROOT/'ml/checkpoints/phase8_xgboost.joblib'),df)
            for target in TARGETS:
                y=df[target].to_numpy(); e=np.abs(xp[target]-y)
                out.append({"implementation":"xgboost_model","target":target,"mae":float(e.mean()),"rmse":float(np.sqrt(np.mean((xp[target]-y)**2))),"status":"measured"})
    except Exception as exc:
        out.append({"implementation":"ml_accuracy","status":"not_run","reason":str(exc)})
    return out


def benchmark_study():
    results={"study":"F1 GPU Race Simulator ML Systems Performance Study","system":_system(),"simulation":benchmark_simulation(),"inference":benchmark_inference(),"accuracy":benchmark_accuracy(),"realtime":benchmark_realtime(),"historical_optimization_ablation":historical_ablation(),"notes":["Missing GPU/Triton/ONNX/TensorRT measurements are represented as not_run; no values are fabricated.","Historical CUDA-simulator results are explicitly separated from real GPU measurements."]}
    return results


def main():
    out=ROOT/"results"; out.mkdir(exist_ok=True)
    data=benchmark_study()
    (out/"benchmark_results.json").write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data,indent=2))

if __name__=="__main__": main()
