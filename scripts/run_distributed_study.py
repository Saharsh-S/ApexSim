"""Run Phases 11-15 reproducibility suite."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from benchmarks.benchmark_distributed import run as distributed
from benchmarks.benchmark_cluster import run as cluster
from benchmarks.benchmark_cost import run as cost
from benchmarks.final_system_benchmark import run as final

def main():
    root=Path("results"); root.mkdir(exist_ok=True)
    outputs={"phase11_distributed":distributed(),"phase12_scheduler":cluster(),"phase14_cost":cost(),"phase15_final":final()}
    Path(root/"phase11_15_study.json").write_text(json.dumps(outputs,indent=2))
    print(json.dumps(outputs,indent=2))
if __name__=="__main__": main()
