"""Phase 11 benchmark: 1/2/4/8 worker scaling using the available backend."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
from examples.basic_race import build_config
from f1sim.models import Strategy
from f1sim.config import TireCompound
from f1sim.distributed import distributed_simulate
from f1sim.gpu.cuda import cuda_backend_status

def run(simulations=256, worker_counts=(1,2,4,8), seed=42):
    config = build_config()
    strategy = Strategy(TireCompound.MEDIUM, [20], [TireCompound.HARD])
    rows=[]; baseline=None
    for n in worker_counts:
        start=perf_counter()
        result=distributed_simulate(config,strategy,simulations,gpu_ids=list(range(n)),seed=seed,baseline_runtime_seconds=baseline,backend="auto")
        elapsed=perf_counter()-start
        if baseline is None: baseline=elapsed
        rows.append({"workers":n,"backend":result.backend,"runtime_seconds":elapsed,"throughput":simulations/elapsed,"speedup":baseline/elapsed,"scaling_efficiency":(baseline/elapsed)/n})
    return {"cuda_status":cuda_backend_status(),"simulations":simulations,"rows":rows}

if __name__ == "__main__":
    out=run()
    Path("results").mkdir(exist_ok=True)
    Path("results/phase11_distributed_scaling.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
