# Profiling Workflow

Phase 4 established the profiling-first workflow. Phase 5 keeps that workflow and adds the Triton hot-kernel comparison.

## NVIDIA tools

When an NVIDIA environment is available:

```bash
nvidia-smi
nsys profile --stats=true python benchmarks/benchmark_gpu.py --sizes 10000 100000
ncu --set full python benchmarks/benchmark_gpu.py --sizes 10000
```

Record:

- kernel duration
- host-to-device and device-to-host transfers
- launch overhead
- GPU utilization
- memory throughput
- achieved occupancy
- registers per thread
- branch divergence

For Triton, use the same workload and device with Nsight Systems/Compute where supported.

## Software timing fallback

`benchmarks/profile_gpu.py` measures CUDA operation boundaries with host timers and reports transfer byte counts. `benchmarks/benchmark_triton.py` uses CUDA events for Triton kernel timing and Numba CUDA events for the CUDA hot kernel.

These fallbacks are useful for runtime comparisons but cannot substitute for Nsight hardware counters such as achieved occupancy or memory throughput.

## Phase 5 measurement rule

Never report CUDA/Triton throughput unless a real NVIDIA device executes both implementations. CUDA simulator timing and CPU timing are retained as development diagnostics only.
