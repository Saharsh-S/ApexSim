# CUDA vs Triton

## Scope

Phase 5 does **not** replace the Phase 3 CUDA race simulator. The CUDA engine remains the full-race GPU implementation. Triton implements the computationally expensive **lap-time arithmetic slice** identified in Phase 4.

The reason for choosing a slice is structural: the race loop contains sequential state transitions (weather, pits, DNF, safety-car state, tire age, fuel, and ordering). Triton is most useful when applied to dense tensor arithmetic inside that loop rather than when forcing the complete state machine into a single translated kernel.

## Selected operation

Phase 4 measured the full CUDA race kernel at 67.56 ms for an 8-race profiling batch in the development environment's CUDA simulator. The kernel performs the same lap-time calculation for every active driver on every lap, making this arithmetic a natural data-parallel target.

The Triton kernel therefore computes:

```text
base pace
+ driver delta
+ car delta
- tire pace
+ degradation
+ warmup
+ fuel effect
+ weather penalty
+ tire/weather mismatch
+ traffic/overtaking penalty
```

It receives the per-race state immediately before lap-time evaluation. Stochastic sampling remains outside this kernel so Triton is not coupled to a particular RNG implementation.

## Triton programming model

### Grid

There is **one Triton program per race**:

```text
program_id(0) -> race
```

Each program processes all drivers in that race using a vector of lanes.

### Block size

`BLOCK_SIZE` is the next power of two greater than or equal to the driver count, capped at 1024. A four-driver race therefore uses four active lanes in an eight-lane block only if configured that way; by default the helper chooses four lanes as the next power of two.

The exact block size is exposed for benchmarking because the best value is hardware dependent.

### Tensor layout

Per-race mutable arrays use row-major `[races, drivers]` storage. Driver and tire configuration arrays are one-dimensional and contiguous.

This gives contiguous accesses across lanes for driver constants and contiguous per-race state accesses for the main state tensors.

### Masking

The driver vector is masked with:

```python
lanes < N_DRIVERS
```

This allows the same kernel to handle arbitrary driver counts without reading or writing inactive lanes.

### Memory access

The kernel reads:

- current accumulated time
- fuel
- tire age
- tire ID
- driver characteristics
- car characteristics
- tire characteristics
- weather mismatch

and writes one `[race, driver]` lap-time tensor. No host/device transfer is required between individual drivers.

## Numerical differences

The CPU/Phase 1 model uses Python/NumPy floating-point arithmetic and the CUDA implementation uses double precision in its race kernel. The Triton hot kernel intentionally uses `float32` for performance-oriented tensor arithmetic.

Therefore exact equality is not expected. Validation uses:

```text
relative tolerance: 2e-4
absolute tolerance: 2e-3 seconds
```

The deterministic hot-kernel slice must remain statistically/numerically equivalent within those tolerances. The full CUDA and CPU race outputs continue to be validated by the Phase 3 tests.

## CPU / CUDA / Triton comparison

The Phase 5 benchmark uses identical generated state tensors for the three hot-kernel implementations. It records mean, median, standard deviation, 5th percentile, 95th percentile, and maximum absolute CPU error.

For full-race Monte Carlo performance, Phase 4 remains the authoritative CPU/NumPy/CUDA benchmark. Phase 5 does not claim that a single Triton lap-time kernel is a complete Triton race simulator.

## Hardware limitation in this environment

The current development environment has:

- no NVIDIA driver
- no `nvidia-smi`
- no CUDA-capable PyTorch device
- no installed Triton package

Consequently, **no CUDA-vs-Triton performance numbers are reported** for this phase. The benchmark records those workloads as `not_run` instead of fabricating values.

The real-device command is:

```bash
python benchmarks/benchmark_triton.py \
    --sizes 10000 100000 1000000 10000000
```

## What can be concluded now?

Nothing definitive about CUDA vs Triton throughput. The project deliberately leaves the comparison open until both kernels run on the same NVIDIA device with the same inputs.

Potential CUDA advantages:

- direct control over the existing race kernel
- explicit thread/block behavior
- mature profiling and debugging tools
- no additional tensor-compiler layer

Potential Triton advantages:

- concise expression of dense tensor arithmetic
- easier experimentation with tile/block shapes
- fewer lines of low-level indexing code
- automatic code generation for supported GPUs

Potential Triton disadvantages:

- the complete race state machine is a poor fit for a first Triton kernel
- compiler-generated register/shared-memory decisions require profiling
- supported operations and hardware behavior can differ by Triton version

The project intentionally does not force a winner.
