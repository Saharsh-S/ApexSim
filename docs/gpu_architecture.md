# GPU Architecture

## Goal

Phase 3 adds a CUDA Monte Carlo backend while keeping the Phase 1 Python simulator as the correctness reference. The GPU implementation is intentionally a first correct baseline, not a fully optimized kernel.

## Mapping

**One CUDA thread = one independent race simulation.**

This mapping is appropriate because Monte Carlo races have minimal cross-simulation dependency. Each race has its own strategy, initial state, random stream, driver state, weather state, tire state, and outputs. The only synchronization required is within one thread's sequential lap loop. Thousands of independent races therefore map naturally to thousands of CUDA threads.

The race itself remains sequential over laps because lap `t+1` depends on the state produced at lap `t`. GPU parallelism is across scenarios, not across laps inside one scenario.

## Host/device memory

### Host → device

For each batch, the host transfers:

- one `int64` seed per simulation
- initial tire IDs
- pit-lap and pit-compound arrays
- driver skill/consistency/aggression/tire-management/qualifying arrays
- car performance/reliability/tire-performance arrays
- tire pace/degradation/warmup arrays
- weather/tire mismatch table
- scalar track/simulation parameters

These are small relative to the number of race scenarios and are reused by all threads. For a typical four-driver race, the seed transfer is `8N` bytes and the final race-time output is `8N` bytes, while positions and pit-stop outputs are each `16N` bytes. Fastest-lap output is another `32N` bytes and safety-car/DNF outputs are `4N` bytes each. Thus the final metric payload is about `72N` bytes for four drivers, excluding the small shared configuration arrays.

### Device-local state

Mutable per-race state stays on the GPU:

- accumulated driver race times
- fuel
- tire age
- current compound
- DNF state
- pit counters
- fastest lap
- positions
- weather
- safety-car state
- RNG state

This avoids transferring intermediate lap state back to the CPU.

### Device → host

Only compact final outputs are copied back:

- race time
- finishing positions
- pit stops
- DNF
- fastest lap
- safety-car laps

There is no host round-trip per lap.

## Random-number strategy

Each simulation receives a distinct deterministic seed derived from the requested batch seed. The kernel uses a per-thread xorshift64* state. The state is initialized independently for each race and mutated only by that race's CUDA thread.

This avoids a shared global RNG and race conditions. Re-running the same GPU batch with the same seeds produces the same GPU results on a fixed CUDA execution environment.

GPU RNG sequences are intentionally **not** expected to match NumPy's `default_rng` bit-for-bit. Therefore CPU/GPU correctness is evaluated by shared deterministic configurations and statistical equivalence rather than pretending different RNG implementations are identical.

## CPU reference relationship

The kernel mirrors the Phase 1 equations and ordering:

1. weather transition
2. safety-car event sampling
3. pit stops
4. pre-lap position ranking
5. DNF sampling
6. driver/car/tire/fuel/weather/traffic lap-time terms
7. stochastic noise
8. safety-car lap-time compression
9. fuel/tire-age updates
10. post-lap ranking
11. safety-car state update

The CPU implementation remains the source of truth. Any future kernel optimization should first be validated against this behavior.

## Correctness strategy

Exact CPU/GPU equality is generally unavailable because NumPy's random generator and the GPU xorshift stream are different generators. Floating-point execution can also differ. Phase 3 therefore provides:

- deterministic GPU replay checks
- deterministic CPU replay checks
- fixed/no-noise configuration comparisons for core race metrics
- statistical equivalence checks over batches

The tolerance and statistical tests are documented in the test suite.

## Benchmarking

`benchmarks/benchmark_gpu.py` reports actual measurements only. It does not fabricate GPU numbers. If no CUDA device is available, the benchmark explicitly reports that the GPU measurement was skipped.

The intended workload tiers are:

- 10K
- 100K
- 1M
- 10M

A warm-up launch is performed before GPU timing so Numba JIT compilation is not mixed into steady-state kernel timing.

## Profiling

The preferred profiling workflow is:

```bash
nsys profile -o profiles/f1sim_gpu python benchmarks/benchmark_gpu.py --simulations 1000000
```

and, where supported:

```bash
ncu --set full --target-processes all python benchmarks/benchmark_gpu.py --simulations 1000000
```

Record kernel runtime, launch overhead, occupancy, achieved bandwidth, and the dominant bottleneck in a profiling report before optimization.

## Expected first bottleneck

The baseline kernel performs a sequential lap loop and uses per-thread arrays for driver state. The first likely bottlenecks are register/local-memory pressure and branch-heavy ranking/traffic logic, rather than global-memory bandwidth. This is intentionally left for profiling rather than optimized by assumption.
