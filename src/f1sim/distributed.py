"""Distributed Monte Carlo execution across multiple GPU workers.

The real backend uses one process per GPU and partitions independent Monte Carlo
seeds across devices. When CUDA is unavailable, the same partition/aggregation
protocol can be exercised with the existing NumPy reference engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from time import perf_counter
from typing import Sequence

import numpy as np

from .config import SimulationConfig
from .models import Strategy
from .monte_carlo import MonteCarloRace, StrategyStatistics, aggregate_statistics, simulate_strategy
from .gpu.cuda import cuda_available, simulate_strategy_gpu


@dataclass(frozen=True)
class WorkerResult:
    worker_id: int
    device_id: int | None
    simulations: int
    runtime_seconds: float
    races: tuple[MonteCarloRace, ...]
    backend: str
    failed: bool = False
    error: str | None = None


@dataclass(frozen=True)
class DistributedResult:
    strategy_id: str
    requested_gpus: int
    active_workers: int
    simulations: int
    runtime_seconds: float
    throughput: float
    scaling_speedup: float
    scaling_efficiency: float
    backend: str
    worker_results: tuple[WorkerResult, ...]
    statistics: StrategyStatistics


def _gpu_worker(payload: tuple[int, int, SimulationConfig, Strategy, int, int, int]) -> WorkerResult:
    worker_id, device_id, config, strategy, simulations, seed, requested = payload
    start = perf_counter()
    try:
        from .monte_carlo import MonteCarloRace
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        batch = simulate_strategy_gpu(config, strategy, simulations, seed)
        races: list[MonteCarloRace] = []
        for j in range(simulations):
            positions = tuple(int(x) if int(x) > 0 else None for x in batch.finishing_positions[j])
            winner = None
            for i, pos in enumerate(positions):
                if pos == 1:
                    winner = config.drivers[i].name
                    break
            races.append(MonteCarloRace(
                seed=int(batch.seeds[j]), winner=winner, finishing_position=positions,
                total_race_time=float(batch.race_times[j]),
                fastest_lap=float(np.min(batch.fastest_laps[j])) if np.any(batch.fastest_laps[j] >= 0) else None,
                pit_stops=int(np.sum(batch.pit_stops[j])), tire_usage=tuple(),
                safety_car_laps=int(batch.safety_car_laps[j]), dnf=bool(batch.dnf[j]),
                total_fuel_used=float(len(config.drivers)), strategy_id="",
            ))
        return WorkerResult(worker_id, device_id, simulations, perf_counter()-start, tuple(races), "cuda")
    except Exception as exc:
        return WorkerResult(worker_id, device_id, simulations, perf_counter()-start, tuple(), "cuda", True, repr(exc))


def _cpu_worker(payload: tuple[int, SimulationConfig, Strategy, int, int]) -> WorkerResult:
    worker_id, config, strategy, simulations, seed = payload
    start = perf_counter()
    try:
        result = simulate_strategy(config, strategy, simulations, seed)
        return WorkerResult(worker_id, None, simulations, perf_counter()-start, result.races, "numpy")
    except Exception as exc:
        return WorkerResult(worker_id, None, simulations, perf_counter()-start, tuple(), "numpy", True, repr(exc))


def _partition(total: int, workers: int) -> list[int]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    base, rem = divmod(total, workers)
    return [base + (i < rem) for i in range(workers)]


def distributed_simulate(
    config: SimulationConfig,
    strategy: Strategy,
    simulations: int,
    gpu_ids: Sequence[int] | None = None,
    seed: int = 0,
    baseline_runtime_seconds: float | None = None,
    backend: str = "auto",
    max_retries: int = 1,
) -> DistributedResult:
    """Run independent simulations across GPUs and aggregate them deterministically.

    ``backend='auto'`` selects CUDA when all requested devices are available;
    otherwise it uses the NumPy reference backend so orchestration is testable
    on machines without GPUs. The fallback is never presented as GPU speed.
    """
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    devices = list(gpu_ids) if gpu_ids is not None else [0]
    if not devices:
        raise ValueError("at least one worker/device is required")
    if backend not in {"auto", "cuda", "numpy"}:
        raise ValueError("backend must be auto, cuda, or numpy")
    use_cuda = backend == "cuda" or (backend == "auto" and cuda_available())
    if use_cuda and not cuda_available():
        if backend == "cuda":
            raise RuntimeError("CUDA backend requested but no CUDA device is available")
        use_cuda = False
    workers = len(devices)
    counts = _partition(simulations, workers)
    start = perf_counter()
    ctx = get_context("spawn")
    results: list[WorkerResult] = []
    if use_cuda:
        payloads = [(i, dev, config, strategy, counts[i], seed + sum(counts[:i]), workers) for i, dev in enumerate(devices)]
        with ctx.Pool(workers) as pool:
            results = list(pool.map(_gpu_worker, payloads))
    else:
        payloads = [(i, config, strategy, counts[i], seed + sum(counts[:i])) for i in range(workers)]
        with ctx.Pool(workers) as pool:
            results = list(pool.map(_cpu_worker, payloads))
    # Retry failed partitions sequentially. This makes recovery deterministic and
    # keeps a failed worker from invalidating otherwise completed work.
    for idx, result in enumerate(results):
        if not result.failed or max_retries <= 0:
            continue
        for _ in range(max_retries):
            count = result.simulations
            retry = (_gpu_worker((result.worker_id, result.device_id or 0, config, strategy, count, seed + sum(counts[:idx]), workers))
                     if use_cuda else _cpu_worker((result.worker_id, config, strategy, count, seed + sum(counts[:idx]))))
            results[idx] = retry
            if not retry.failed:
                break
    races = tuple(race for worker in sorted(results, key=lambda x: x.worker_id) for race in worker.races)
    if len(races) != simulations:
        errors = "; ".join(r.error or "unknown worker error" for r in results if r.failed)
        raise RuntimeError(f"distributed run produced {len(races)}/{simulations} races: {errors}")
    sid = races[0].strategy_id
    if not sid:
        from .monte_carlo import strategy_id
        sid = strategy_id(strategy)
        races = tuple(MonteCarloRace(**{**r.__dict__, "strategy_id": sid}) for r in races)
    elapsed = perf_counter() - start
    throughput = simulations / elapsed if elapsed else float("inf")
    baseline = baseline_runtime_seconds if baseline_runtime_seconds is not None else elapsed
    speedup = baseline / elapsed if elapsed else float("inf")
    efficiency = speedup / workers if workers else 0.0
    stats = aggregate_statistics(races, sid, simulations)
    return DistributedResult(sid, len(devices), workers, simulations, elapsed, throughput, speedup, efficiency,
                             "cuda" if use_cuda else "numpy-fallback", tuple(results), stats)
