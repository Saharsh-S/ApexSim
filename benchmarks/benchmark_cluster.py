from __future__ import annotations
import json
from pathlib import Path
from f1sim.cluster import run_cluster
from f1sim.scheduler import GPUJob, GPUResource

def run():
    jobs=[GPUJob(f"job-{i}", "simulation" if i%2==0 else "inference", priority=i%3, required_memory_gb=1, retries=1) for i in range(8)]
    resources=[GPUResource(i, memory_gb=16) for i in range(4)]
    def handler(job,gpu): return {"gpu":gpu,"workload":job.workload}
    result=run_cluster(jobs,resources,{"simulation":handler,"inference":handler})
    return {"jobs":result.jobs,"completed":result.completed,"failed":result.failed,"runtime_seconds":result.runtime_seconds,"assignments":result.assignments}
if __name__=="__main__":
    Path("results").mkdir(exist_ok=True); data=run(); Path("results/phase12_scheduler.json").write_text(json.dumps(data,indent=2)); print(json.dumps(data,indent=2))
