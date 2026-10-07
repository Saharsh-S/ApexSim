from __future__ import annotations
import json
from pathlib import Path
from f1sim.fault_tolerance import Checkpoint, deterministic_seed

def run():
    cp=Checkpoint("mc-10m",((0,1000),(2000,3000)),(3,),{"seed":42})
    path=Path("results/phase13_checkpoint.json"); cp.save(path); restored=Checkpoint.load(path)
    return {"job_id":restored.job_id,"completed_ranges":restored.completed_ranges,"missing_ranges":restored.missing_ranges(4000),"replay_seeds":[deterministic_seed(42,i) for i in range(5)]}
if __name__=="__main__":
    Path("results").mkdir(exist_ok=True); print(json.dumps(run(),indent=2))
