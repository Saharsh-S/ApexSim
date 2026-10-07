# Distributed GPU Simulation (Phase 11)

The Monte Carlo workload is embarrassingly parallel: each simulation has an independent seed. Phase 11 partitions a fixed simulation index range across workers, runs one process per real CUDA device when available, and aggregates outcomes deterministically.

## Scaling metric

For `N` workers:

`speedup(N) = T(1) / T(N)`

`efficiency(N) = speedup(N) / N`

A 4x speedup on four GPUs is 100% scaling efficiency. Lower efficiency indicates overheads such as process startup, host scheduling, memory transfers, synchronization, load imbalance, or shared-resource contention.

If CUDA is unavailable, the same partitioning protocol runs on the NumPy reference backend. Those results are explicitly labeled `numpy-fallback` and must not be interpreted as GPU scaling.
