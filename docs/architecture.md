# Monte Carlo Architecture

## Purpose

Phase 2 turns the Phase 1 single-race simulator into an independent-scenario workload. The goal is not to make the F1 model more physically detailed. The goal is to make the computational structure representative of a future GPU workload.

## Source of truth

`f1sim.simulate_race()` remains the correctness reference. The straightforward Monte Carlo engine calls it once per seed. This means Phase 2 does not silently introduce a second CPU race model.

The NumPy engine mirrors the Phase 1 equations in batched arrays. It is an execution baseline, not a new source of truth. Its results are intentionally compact because retaining every per-lap state for hundreds of thousands of races would distort the memory benchmark.

## Computational structure

```text
Strategy
   |
   +--> seed sequence
          |
          +--> independent race 0
          +--> independent race 1
          +--> independent race 2
          +--> ...
          +--> independent race N

N races x M strategies
          |
          v
   Monte Carlo outcomes
          |
          v
   aggregate statistics
          |
          +--> ranking
          +--> convergence
          +--> risk / reward
```

Each race has its own random stream state and does not depend on the result of another race. This is the key property we want to preserve when the workload moves to GPU kernels.

## Python baseline

`simulate_strategies_python()` uses `simulate_race()` directly. It is deliberately straightforward and provides the most transparent CPU timing baseline.

## NumPy baseline

`simulate_strategies_numpy()` stores race and driver state in arrays shaped approximately as:

```text
[simulation, driver]
```

Lap operations are then performed across all simulations simultaneously. Random noise, failures, weather transitions, tire degradation, fuel effects, safety-car compression, and traffic calculations are batched where practical.

This establishes a second CPU baseline without introducing multiprocessing or native extensions.

## Why this maps to GPUs

A future CUDA/Triton implementation can map the independent simulation dimension to threads/program instances:

```text
GPU grid
  |
  +-- simulation 0
  +-- simulation 1
  +-- simulation 2
  +-- ...
  +-- simulation N
```

Within each simulation, drivers and laps create additional parallelizable dimensions. The major synchronization boundary is the race lap because position ordering and safety-car state can affect the next lap.

The ideal future implementation will therefore keep global simulation state contiguous, minimize host/device transfers, use deterministic seed derivation, and benchmark throughput and memory bandwidth separately.

## Benchmark methodology

Benchmark inputs must remain fixed when comparing implementations:

- same track configuration
- same driver/car configuration
- same strategy set
- same simulation count
- same seed sequence
- same Python version and machine when possible

The benchmark reports elapsed time, races per second, and traced peak Python memory where available. `tracemalloc` is a Python allocation measurement, not a complete process RSS measurement.

## Phase 3 boundary

CUDA, Triton, GPU kernels, GPU memory profiling, and ML-based strategy optimization are intentionally not included in Phase 2.
