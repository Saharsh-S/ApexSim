# Phase 4 Performance Analysis

## Scope

Phase 4 is a performance-engineering pass over the Phase 3 CUDA implementation. No new F1 behavior was introduced. The CPU simulator remains the correctness reference.

A major environment limitation is important: the development container exposes **no NVIDIA GPU, CUDA driver, `nvcc`, Nsight Systems, Nsight Compute, or `nvidia-smi`**. Therefore hardware occupancy, achieved bandwidth, register count, branch efficiency, GPU utilization, and real CUDA kernel latency cannot be measured here. The repository records those metrics as unavailable rather than fabricating values.

For software-level evidence, the CUDA kernel was executed with `NUMBA_ENABLE_CUDASIM=1`. CUDA-simulator timings are useful for validating control flow and comparing algorithmic variants, but they are **not GPU performance measurements**.

## Baseline workload measurements

Actual CPU/NumPy measurements from this environment:

| Simulations | CPU | NumPy | CUDA |
|---:|---:|---:|---:|
| 10K | 5.7130 s / 1,750 sims/s | 0.4628 s / 21,607 sims/s | Not run: no CUDA device |
| 100K | Not run: >300 s validation limit | 4.8295 s / 20,706 sims/s | Not run: no CUDA device |
| 1M | Not run | Not run: >180 s bounded run | Not run |
| 10M | Not run | Not run | Not run |

These are real measurements where execution completed. Missing cells are explicitly marked as not run.

## Bottleneck 1: Kernel work dominates the end-to-end path

### What was measured

With 8 simulations under the CUDA simulator:

- Kernel: ~72.30 ms
- Host-to-device: ~0.123 ms
- Device-to-host: ~0.038 ms

The kernel therefore accounts for approximately 99.5% of measured wall time in that software-simulation run.

### Why it matters

Reducing transfer overhead cannot compensate for a kernel that performs most of the work. The first optimization target should therefore be the per-race kernel itself.

### Evidence

The same profiling run reports 1,440 ranking comparisons and an estimated 3,360 RNG draws for eight four-driver races.

### Proposed solution

Reduce redundant per-lap work and avoid unnecessary RNG/math operations when their configured effect is zero.

### Result

The deterministic configuration optimization reduced CUDA-simulator kernel time from **74.37 ms to 44.71 ms** in the measured before/after run, a **39.9% reduction**. Three post-change repeats were 46.61 ms, 44.04 ms, and 44.71 ms.

This is a software-simulator result, not a claim about real GPU speedup.

## Bottleneck 2: Race-position sorting is O(D²) inside every race

### What was measured

The kernel performs selection-style ranking over the driver list. With four drivers and 30 laps, the measured current kernel performs:

- 180 comparisons per race
- 1,440 comparisons for 8 races

A driver-scaling probe produced:

| Drivers | Kernel time, 8 races | Ranking comparisons |
|---:|---:|---:|
| 1 | 34.67 ms | 0 |
| 2 | 42.55 ms | 30/race |
| 4 | 69.63 ms | 180/race |

This establishes that ranking work grows with driver count and is a real component of the kernel workload.

### Why it matters

Each race is a sequential state machine, so ranking is performed inside every race rather than as a separate global GPU operation. The quadratic ranking cost therefore consumes per-thread work.

### Proposed solution

The implementation now avoids a redundant **pre-lap ranking** when no pit stop changes accumulated race time. The post-lap ranking is already current at the next lap. This cuts the ranking work for ordinary laps in half while preserving the Phase 1 ordering semantics.

### Result

The algorithmic comparison count was reduced from the previous two-rankings-per-lap structure to one post-lap ranking plus only the required pit-stop re-rank. The CUDA simulator did not show a trustworthy wall-time improvement from this change, so no percentage speedup is claimed for it.

## Bottleneck 3: Random-number generation and normal-noise math

### What was measured

The baseline race model requires stochastic draws for weather, safety cars, DNF checks, and lap-time noise. The current instrumentation estimates **420 RNG draws per four-driver 30-lap race**, or 3,360 for 8 races.

A controlled deterministic run disables all stochastic effects. Before the conditional RNG optimization its CUDA-simulator kernel time was 74.37 ms. After avoiding RNG/normal generation when the configured probability/noise is exactly zero, repeated runs were 46.61 ms, 44.04 ms, and 44.71 ms.

### Why it matters

The Box-Muller normal generator uses logarithm, square-root, and cosine operations. Performing those calculations when `stochastic_noise_std == 0` is pure wasted work.

### Proposed solution

Guard stochastic operations with configuration checks:

- skip weather RNG when transition probability is zero
- skip safety-car RNG when probability is zero
- skip DNF RNG when failure probability is zero
- skip normal generation when noise scale is zero

### Result

The measured deterministic case improved from **74.37 ms to 44.71 ms**, a **39.9% reduction** in CUDA-simulator kernel time.

For normal stochastic races, the guards preserve the existing model and remain active because their configured probabilities are non-zero.

## Optimization 2: Persistent immutable device configuration

Repeated strategy simulations previously re-uploaded immutable driver/car/tire arrays on every strategy launch. Phase 4 adds `GPUDeviceContext` / `create_gpu_context()` so those arrays can remain resident on the device.

Measured CUDA-simulator transfer component:

- Before: 1,536 bytes of host-to-device input
- After persistent context: 1,072 bytes
- Payload reduction: **30.2%**

The measured host-to-device component fell from approximately 72.8 µs to 20.0 µs in the software simulator. This is a transfer-component result, not a real GPU throughput claim.

The optimization matters most for workloads with many strategy launches. It does not eliminate the per-batch seed and strategy-specific pit data that must still change.

## What was *not* identified as a bottleneck

Host/device transfer is not the dominant path for this workload. The measured CUDA-simulator transfer times are orders of magnitude smaller than kernel execution. Nevertheless, persistent device context was implemented because it is effectively free for correctness and becomes more valuable as strategy count grows.

## Profiling limitations

The following hardware measurements remain pending on a real NVIDIA system:

- GPU utilization
- SM occupancy
- achieved memory bandwidth
- register count / register pressure
- local-memory spills
- branch efficiency / divergence
- actual kernel launch latency
- PCIe/NVLink transfer throughput
- power/utilization telemetry

Use Nsight Systems and Nsight Compute as the authoritative source for those measurements.

## Engineering conclusion

The correct optimization order is:

```text
CPU reference
    ↓
Basic CUDA implementation
    ↓
Measure
    ↓
Reduce wasted per-thread work
    ↓
Keep immutable data resident
    ↓
Measure again on real GPU hardware
    ↓
Only then tune block size, memory layout, registers, and occupancy
```

Phase 4 deliberately stops before architecture-specific micro-optimization because the required NVIDIA hardware is unavailable in the development environment.
