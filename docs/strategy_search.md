# Phase 6: GPU-Powered Strategy Search

## Goal

Phase 6 adds an automatic strategy-search layer on top of the existing Phase 1 CPU simulator, Phase 2 Monte Carlo engine, and Phase 3 CUDA backend. It does **not** introduce a second F1 race model.

The search space is generated procedurally from valid tire sequences and pit windows. Each candidate also carries two search dimensions: target pace and risk level. These are retained as candidate metadata in this baseline; the existing Phase 1 race equations remain authoritative and are not silently modified.

## Candidate generation

`StrategySearchConfig` controls:

- candidate count
- simulations per candidate
- minimum/maximum stops
- minimum segment length
- random seed
- target-pace range
- risk range

Candidates are sampled from all valid Soft/Medium/Hard strategy skeletons. Adjacent identical compounds and impossible pit windows are rejected.

This means 100,000 candidates are not a manually maintained list. The generator samples the combinatorial strategy space deterministically from a seed.

## Parallel workload

The target workload is:

```text
candidate strategies × independent simulations
```

For example:

```text
100,000 candidates × 1,000 simulations
= 100,000,000 race futures
```

### Current GPU mapping

The existing Phase 3 CUDA kernel maps:

```text
1 CUDA thread = 1 race simulation
```

Phase 6 keeps that mapping for correctness. The host schedules candidates and the GPU executes all simulations belonging to each candidate in parallel.

This is deliberately conservative: it reuses the tested Phase 3 race kernel rather than introducing a second GPU race implementation. The next natural GPU optimization would flatten `(candidate, simulation)` into one CUDA grid and store strategy parameters in device arrays, eliminating candidate-by-candidate launches. That is future optimization work, not a new simulation model.

### CPU mapping

The Python CPU baseline evaluates candidates sequentially using the Phase 1 simulator. The NumPy baseline batches the simulations belonging to each candidate.

## Outputs

The search returns the top candidates with:

- expected finishing position
- win probability
- podium probability
- expected race time
- risk
- expected tire usage
- expected pit stops
- total search time
- strategies/sec
- simulations/sec

## Ranking

The default ranking score rewards win probability and podium probability while penalizing race-time risk and candidate risk. Ranking is intentionally configurable at the analysis layer rather than embedded in the race simulator.

## GPU/CPU correctness

CPU and GPU use the same Phase 1/Phase 3 race rules for the evaluated strategy. GPU compact outputs are converted into the same `MonteCarloResult` representation used by the CPU engines.

Because the CUDA backend has its own random stream, stochastic race values are not expected to be bit-identical to the CPU RNG. Deterministic no-randomness configurations are used for numerical checks; stochastic searches should be compared statistically.

## Scale policy

Search size is configurable. The command-line example supports million-scale candidate workloads, but the default examples use a much smaller count so they remain practical on CPU-only machines. A real GPU should be used before attempting 100 million futures.

## Visualization

Phase 6 adds plots for:

1. candidate strategy space
2. expected performance
3. risk/reward
4. top strategies

All plots are generated with Matplotlib and can be saved without introducing a web dashboard.

## Target pace and risk semantics

`target_pace` and `risk_level` are explicit search dimensions, but they are not secretly injected into the Phase 1 race equations. This is intentional: Phase 6 is a search-engine experiment, and changing the race model just to make new knobs affect the score would break comparability with earlier phases.

The baseline search therefore evaluates each tire/pit strategy with the existing simulator and uses risk as a ranking preference. Target pace is retained in the candidate identity and search-space visualization so a future strategy-conditioned race kernel can consume it without changing the search API.

## Scale example

```bash
python benchmarks/benchmark_strategy_search.py --candidates 100000 --simulations 1000 --backend cuda
```

This requests 100 million simulated futures. The command will fail clearly if no CUDA device is available rather than silently falling back to a potentially impractical CPU job.
