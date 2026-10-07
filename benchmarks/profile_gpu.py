"""Profile-level instrumentation for the CUDA race kernel.

On a real NVIDIA GPU, use Nsight Systems/Compute as the authoritative source
for kernel, occupancy, bandwidth, and launch metrics. Without hardware this
script can run under NUMBA_ENABLE_CUDASIM=1 to collect software-level timing and
operation-count evidence only.
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
from f1sim import Strategy, TireCompound
from f1sim.gpu import cuda_backend_status, simulate_strategy_gpu


def run(label, config, strategy, simulations):
    start = time.perf_counter()
    result = simulate_strategy_gpu(config, strategy, simulations, seed=42, return_timing=True)
    elapsed = time.perf_counter() - start
    timing = result.timing
    return {
        "label": label,
        "simulations": simulations,
        "wall_seconds": elapsed,
        "input_encoding_seconds": timing.input_encoding_seconds,
        "host_to_device_seconds": timing.host_to_device_seconds,
        "kernel_seconds": timing.kernel_seconds,
        "device_to_host_seconds": timing.device_to_host_seconds,
        "host_to_device_bytes": timing.host_to_device_bytes,
        "device_to_host_bytes": timing.device_to_host_bytes,
        "sort_comparisons": timing.sort_comparisons,
        "estimated_rng_draws": timing.estimated_rng_draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--output", default="benchmarks/results/phase4_profile.json")
    args = parser.parse_args()
    config = build_config()
    strategy = Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD])
    if "NUMBA_ENABLE_CUDASIM" not in os.environ and "CUDA device unavailable" in cuda_backend_status():
        print("Real CUDA unavailable. Set NUMBA_ENABLE_CUDASIM=1 for software-level profiling.")
        raise SystemExit(0)
    data = {"cuda_status": cuda_backend_status(), "measurements": [run("four_driver_full", config, strategy, args.simulations)]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
