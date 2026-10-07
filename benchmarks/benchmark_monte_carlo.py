"""Benchmark the Python and NumPy Monte Carlo CPU baselines."""

from __future__ import annotations

import argparse
import os
import sys
import tracemalloc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.basic_race import build_config
from f1sim import Strategy, TireCompound
from f1sim.monte_carlo import benchmark_engine, simulate_strategies_numpy, simulate_strategies_python


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=1000)
    args = parser.parse_args()
    if args.strategies <= 0 or args.simulations <= 0:
        raise SystemExit("--strategies and --simulations must be positive")

    config = build_config()
    candidates = [
        Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD]),
        Strategy(TireCompound.SOFT, [10], [TireCompound.MEDIUM]),
        Strategy(TireCompound.SOFT, [10, 22], [TireCompound.MEDIUM, TireCompound.HARD]),
        Strategy(TireCompound.MEDIUM, [12, 23], [TireCompound.HARD, TireCompound.MEDIUM]),
    ]
    strategies = [candidates[i % len(candidates)] for i in range(args.strategies)]
    total = args.strategies * args.simulations
    print(f"Strategies: {args.strategies}")
    print(f"Simulations/strategy: {args.simulations:,}")
    print(f"Total simulations: {total:,}\n")

    tracemalloc.start()
    python_result = benchmark_engine(simulate_strategies_python, config, strategies, args.simulations, seed=42)
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Python runtime: {python_result.runtime_seconds:.4f} seconds")
    print(f"Python races/sec: {python_result.races_per_second:,.1f}")
    print(f"Python peak traced memory: {peak_python / 1024 / 1024:.2f} MB\n")

    tracemalloc.start()
    numpy_result = benchmark_engine(simulate_strategies_numpy, config, strategies, args.simulations, seed=42)
    _, peak_numpy = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"NumPy runtime: {numpy_result.runtime_seconds:.4f} seconds")
    print(f"NumPy races/sec: {numpy_result.races_per_second:,.1f}")
    print(f"NumPy peak traced memory: {peak_numpy / 1024 / 1024:.2f} MB")
    print(f"\nSpeedup: {python_result.runtime_seconds / numpy_result.runtime_seconds:.2f}x")


if __name__ == "__main__":
    main()
