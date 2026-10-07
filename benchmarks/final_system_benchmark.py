"""Phase 15 consolidated benchmark. GPU rows are marked not_run when unavailable."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
from examples.basic_race import build_config
from f1sim.config import TireCompound
from f1sim.models import Strategy
from f1sim.monte_carlo import simulate_strategies_python, simulate_strategies_numpy
from f1sim.gpu.cuda import cuda_available, cuda_backend_status

def run(simulations=256):
    config=build_config(); strategy=Strategy(TireCompound.MEDIUM,[20],[TireCompound.HARD])
    rows=[]
    for name,fn in [("Python CPU",simulate_strategies_python),("NumPy CPU",simulate_strategies_numpy)]:
        t=perf_counter(); fn(config,[strategy],simulations,42); elapsed=perf_counter()-t
        rows.append({"system":name,"status":"measured","runtime_seconds":elapsed,"throughput":simulations/elapsed})
    status=cuda_backend_status()
    for name in ["CUDA baseline","Optimized CUDA","Triton","Multi-GPU"]:
        rows.append({"system":name,"status":"not_run","runtime_seconds":None,"throughput":None,"reason":status})
    return {"simulations":simulations,"cuda_status":status,"rows":rows}
if __name__=="__main__":
    Path("results").mkdir(exist_ok=True); data=run(); Path("results/phase15_final_benchmark.json").write_text(json.dumps(data,indent=2)); print(json.dumps(data,indent=2))
