#!/usr/bin/env python3
"""Run the complete reproducible performance study, including Phases 11-15."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from benchmarks.performance_study import benchmark_study
from benchmarks.benchmark_distributed import run as distributed
from benchmarks.benchmark_cluster import run as cluster
from benchmarks.benchmark_fault_tolerance import run as fault
from benchmarks.benchmark_cost import run as cost
from benchmarks.final_system_benchmark import run as final

def main():
    results_dir=ROOT/'results'; results_dir.mkdir(exist_ok=True)
    outputs={"phase1_10": benchmark_study(), "phase11_distributed": distributed(), "phase12_scheduler": cluster(), "phase13_fault_tolerance": fault(), "phase14_cost": cost(), "phase15_final": final()}
    (results_dir/'all_benchmarks.json').write_text(json.dumps(outputs,indent=2)+'\n')
    try:
        from benchmarks.generate_study_artifacts import generate
        generate(outputs["phase1_10"], results_dir)
    except Exception as exc:
        (results_dir/'artifact_generation_warning.txt').write_text(str(exc))
    print(json.dumps(outputs,indent=2))

if __name__=='__main__': main()
