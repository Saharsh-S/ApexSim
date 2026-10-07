"""Measured CPU/NumPy/CUDA benchmark for the F1 Monte Carlo engine.

GPU results are emitted only when a real CUDA device is available. The script
also supports chunked NumPy execution so large workloads do not require the
entire Monte Carlo state to fit in RAM at once.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.basic_race import build_config
from f1sim import Strategy, TireCompound, create_gpu_context
from f1sim.gpu import cuda_available, cuda_backend_status, simulate_strategy_gpu
from f1sim.monte_carlo import simulate_strategies_numpy, simulate_strategies_python


def timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def run_chunked(fn, config, strategy, simulations, seed, chunk_size):
    remaining = simulations
    current_seed = seed
    start = time.perf_counter()
    while remaining:
        count = min(remaining, chunk_size)
        fn(config, [strategy], count, seed=current_seed)
        remaining -= count
        current_seed += count
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--backends", nargs="+", choices=("cpu", "numpy", "gpu"), default=("cpu", "numpy", "gpu"))
    parser.add_argument("--results", default="benchmarks/results/phase4_benchmark.json")
    args = parser.parse_args()

    sizes = args.sizes or ([args.simulations] if args.simulations else [10_000, 100_000, 1_000_000, 10_000_000])
    if any(x <= 0 for x in sizes) or args.chunk_size <= 0:
        raise SystemExit("simulation counts and chunk size must be positive")

    config = build_config()
    strategy = Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD])
    gpu_ok = cuda_available()
    print(f"CUDA: {cuda_backend_status()}")

    records = []
    gpu_context = None
    if gpu_ok and "gpu" in args.backends:
        gpu_context = create_gpu_context(config)
        simulate_strategy_gpu(config, strategy, min(32, min(sizes)), seed=42, threads_per_block=args.threads, context=gpu_context)

    for simulations in sizes:
        print(f"\nSimulations: {simulations:,}")
        row = {"simulations": simulations, "cpu": None, "numpy": None, "gpu": None}

        if "cpu" in args.backends:
            start = time.perf_counter()
            simulate_strategies_python(config, [strategy], simulations, seed=42)
            elapsed = time.perf_counter() - start
            row["cpu"] = {"runtime_seconds": elapsed, "simulations_per_second": simulations / elapsed}
            print(f"CPU   {elapsed:.4f}s  {simulations / elapsed:,.1f} sims/s")

        if "numpy" in args.backends:
            elapsed = run_chunked(simulate_strategies_numpy, config, strategy, simulations, 42, args.chunk_size)
            row["numpy"] = {"runtime_seconds": elapsed, "simulations_per_second": simulations / elapsed}
            print(f"NumPy {elapsed:.4f}s  {simulations / elapsed:,.1f} sims/s")

        if "gpu" in args.backends:
            if not gpu_ok:
                row["gpu"] = {"status": "not_run", "reason": cuda_backend_status()}
                print("GPU   not run: no CUDA device/driver available")
            else:
                result = simulate_strategy_gpu(config, strategy, simulations, seed=42, threads_per_block=args.threads, return_timing=True, context=gpu_context)
                timing = result.timing
                row["gpu"] = {
                    "runtime_seconds": timing.total_seconds,
                    "kernel_seconds": timing.kernel_seconds,
                    "host_to_device_seconds": timing.host_to_device_seconds,
                    "device_to_host_seconds": timing.device_to_host_seconds,
                    "simulations_per_second": simulations / timing.total_seconds,
                    "gpu_memory_bytes": timing.host_to_device_bytes + timing.device_to_host_bytes,
                    "host_to_device_bytes": timing.host_to_device_bytes,
                    "device_to_host_bytes": timing.device_to_host_bytes,
                }
                print(f"GPU   {timing.total_seconds:.4f}s  {simulations / timing.total_seconds:,.1f} sims/s")

        if row["cpu"] and row["gpu"] and "runtime_seconds" in row["gpu"]:
            row["gpu_speedup_vs_cpu"] = row["cpu"]["runtime_seconds"] / row["gpu"]["runtime_seconds"]
        if row["numpy"] and row["gpu"] and "runtime_seconds" in row["gpu"]:
            row["gpu_speedup_vs_numpy"] = row["numpy"]["runtime_seconds"] / row["gpu"]["runtime_seconds"]
        records.append(row)

    output = {"cuda_status": cuda_backend_status(), "records": records}
    path = Path(args.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    main()
