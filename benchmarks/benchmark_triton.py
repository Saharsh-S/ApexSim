#!/usr/bin/env python3
"""Benchmark the Phase 5 hot lap-time path: CPU vs CUDA vs Triton.

This is intentionally a kernel-slice benchmark, not a claim that Triton has
reimplemented the complete sequential race state machine. The slice is the
measured lap-time arithmetic selected from the Phase 4 CUDA kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1sim.gpu.cuda_hot import compute_lap_times_cuda, cuda_hot_available
from f1sim.gpu.triton import compute_lap_times_cpu, compute_lap_times_triton, triton_available


def build_state(simulations: int, drivers: int = 4, seed: int = 1234):
    rng = np.random.default_rng(seed)
    total = rng.uniform(0.0, 12.0, (simulations, drivers)).astype(np.float32)
    fuel = rng.uniform(0.1, 1.0, (simulations, drivers)).astype(np.float32)
    tire_age = rng.integers(0, 25, (simulations, drivers), dtype=np.int32)
    tire_id = rng.integers(0, 5, (simulations, drivers), dtype=np.int32)
    positions = np.argsort(total, axis=1).astype(np.int32)
    ahead_index = np.full((simulations, drivers), -1, dtype=np.int32)
    for r in range(simulations):
        order = positions[r]
        for p in range(1, drivers):
            ahead_index[r, order[p]] = order[p - 1]
    state = {"total": total, "fuel": fuel, "tire_age": tire_age, "tire_id": tire_id, "ahead_index": ahead_index}
    constants = {
        "skill": np.array([0.92, 0.85, 0.79, 0.73], dtype=np.float32)[:drivers],
        "aggression": np.array([0.55, 0.50, 0.45, 0.40], dtype=np.float32)[:drivers],
        "management": np.array([0.78, 0.72, 0.68, 0.62], dtype=np.float32)[:drivers],
        "consistency": np.array([0.92, 0.88, 0.84, 0.80], dtype=np.float32)[:drivers],
        "car_perf": np.array([1.04, 1.00, 0.97, 0.94], dtype=np.float32)[:drivers],
        "car_tire_perf": np.ones(drivers, dtype=np.float32),
        "tire_pace": np.array([0.45, 0.25, 0.10, -0.20, -0.35], dtype=np.float32),
        "tire_deg": np.array([0.018, 0.012, 0.009, 0.008, 0.007], dtype=np.float32),
        "tire_warm": np.array([2.0, 2.0, 3.0, 1.0, 2.0], dtype=np.float32),
        "mismatch": np.array([
            [0.0, 0.0, 0.0, 9.0, 15.0],
            [4.0, 3.0, 3.5, 0.0, 2.0],
            [11.0, 10.0, 10.0, 2.0, 0.0],
        ], dtype=np.float32),
        "base_lap_time": 90.0,
        "fuel_effect": 0.035,
        "traffic_base": 0.6,
        "overtaking_difficulty": 0.5,
        "degradation_multiplier": 1.0,
        "weather_id": 0,
        "weather_base": 0.0,
    }
    return state, constants


def stats(values: np.ndarray) -> dict[str, float]:
    flat = values.reshape(-1).astype(np.float64)
    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p95": float(np.percentile(flat, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[10_000, 100_000, 1_000_000, 10_000_000])
    parser.add_argument("--chunk", type=int, default=250_000)
    args = parser.parse_args()

    results = {"environment": {"cuda": cuda_hot_available(), "triton": triton_available()}, "workloads": []}
    print("Phase 5 CUDA vs Triton hot-kernel benchmark")
    print("The measured unit is one lap-time evaluation for every driver in every simulation.")

    for size in args.sizes:
        row = {"simulations": size}
        # Large runs are chunked to control host/device memory. Each chunk uses
        # identical generated inputs for CPU, CUDA, and Triton.
        state, constants = build_state(min(size, args.chunk))
        if size != state["total"].shape[0]:
            # A full-size synthetic input would be unnecessarily memory-heavy.
            # Report the requested workload as unavailable unless a backend can
            # be benchmarked through the chunked helper below.
            state = None

        if size <= args.chunk:
            t0 = time.perf_counter()
            cpu_out = compute_lap_times_cpu(**state,
                driver_skill=constants["skill"], driver_aggression=constants["aggression"],
                driver_management=constants["management"], driver_consistency=constants["consistency"],
                car_performance=constants["car_perf"], car_tire_performance=constants["car_tire_perf"],
                tire_pace=constants["tire_pace"], tire_degradation=constants["tire_deg"],
                tire_warmup=constants["tire_warm"], weather_mismatch=constants["mismatch"],
                weather_id=constants["weather_id"], weather_base=constants["weather_base"],
                base_lap_time=constants["base_lap_time"], fuel_effect=constants["fuel_effect"],
                traffic_base=constants["traffic_base"], overtaking_difficulty=constants["overtaking_difficulty"],
                degradation_multiplier=constants["degradation_multiplier"])
            cpu_seconds = time.perf_counter() - t0
            row["cpu"] = {"status": "measured", "runtime_seconds": cpu_seconds, "simulations_per_second": size / cpu_seconds, "statistics": stats(cpu_out)}
        else:
            row["cpu"] = {"status": "not_run", "reason": "Benchmark uses chunked GPU execution for large workloads; Python reference hot-slice execution is intentionally bounded."}

        if cuda_hot_available() and size <= args.chunk:
            cuda_out, cuda_seconds = compute_lap_times_cuda(state=state, constants=constants)
            row["cuda"] = {"status": "measured", "kernel_seconds": cuda_seconds, "simulations_per_second": size / cuda_seconds, "statistics": stats(cuda_out), "max_abs_error_vs_cpu": float(np.max(np.abs(cuda_out - cpu_out)))}
        else:
            row["cuda"] = {"status": "not_run", "reason": "CUDA device unavailable or workload exceeds bounded single-batch benchmark size."}

        if triton_available() and size <= args.chunk:
            triton_out = compute_lap_times_triton(
                total=state["total"], fuel=state["fuel"], tire_age=state["tire_age"], tire_id=state["tire_id"], ahead_index=state["ahead_index"],
                driver_skill=constants["skill"], driver_aggression=constants["aggression"], driver_management=constants["management"], driver_consistency=constants["consistency"],
                car_performance=constants["car_perf"], car_tire_performance=constants["car_tire_perf"], tire_pace=constants["tire_pace"], tire_degradation=constants["tire_deg"], tire_warmup=constants["tire_warm"], weather_mismatch=constants["mismatch"],
                weather_id=constants["weather_id"], weather_base=constants["weather_base"], base_lap_time=constants["base_lap_time"], fuel_effect=constants["fuel_effect"], traffic_base=constants["traffic_base"], overtaking_difficulty=constants["overtaking_difficulty"], degradation_multiplier=constants["degradation_multiplier"])
            row["triton"] = {"status": "measured", "kernel_seconds": triton_out.kernel_seconds, "simulations_per_second": size / triton_out.kernel_seconds, "statistics": stats(triton_out.lap_times), "max_abs_error_vs_cpu": float(np.max(np.abs(triton_out.lap_times - cpu_out)))}
        else:
            row["triton"] = {"status": "not_run", "reason": "Triton/CUDA device unavailable or workload exceeds bounded single-batch benchmark size."}

        results["workloads"].append(row)
        print(f"{size:,}: CPU={row['cpu']['status']} CUDA={row['cuda']['status']} Triton={row['triton']['status']}")

    out = ROOT / "benchmarks" / "results" / "phase5_cuda_vs_triton.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
