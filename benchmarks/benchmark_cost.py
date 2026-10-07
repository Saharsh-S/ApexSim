"""Phase 14 cost benchmark.

No cloud price or power reading is assumed. Set F1_GPU_DOLLARS_PER_HOUR and
F1_GPU_POWER_WATTS from a real deployment to produce a measured estimate.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from f1sim.cost_model import GPUCostProfile, estimate_cost

def run():
    price=os.getenv("F1_GPU_DOLLARS_PER_HOUR")
    power=os.getenv("F1_GPU_POWER_WATTS")
    if price is None:
        return {"status":"not_run","reason":"No real GPU price supplied; set F1_GPU_DOLLARS_PER_HOUR.","rows":[]}
    profile=GPUCostProfile("user-supplied-gpu",float(price),None if power is None else float(power))
    rows=[]
    # Cost requires a measured runtime. Consume phase-11 CUDA measurements only
    # when they exist; otherwise leave the metric unavailable.
    scaling=Path("results/phase11_distributed_scaling.json")
    if scaling.exists():
        data=json.loads(scaling.read_text())
        for row in data.get("rows",[]):
            if row.get("backend")=="cuda":
                e=estimate_cost(row["runtime_seconds"],data["simulations"],profile)
                rows.append({"workers":row["workers"],"runtime_seconds":row["runtime_seconds"],"dollars":e.dollars,"energy_kwh":e.energy_kwh})
    return {"status":"measured" if rows else "not_run","reason":None if rows else "No measured CUDA runtime available.","rows":rows}
if __name__=="__main__":
    Path("results").mkdir(exist_ok=True); data=run(); Path("results/phase14_cost_model.json").write_text(json.dumps(data,indent=2)); print(json.dumps(data,indent=2))
