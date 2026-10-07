# Final AI Systems Benchmark (Phase 15)

Phase 15 consolidates the CPU, CUDA, Triton, ML inference, multi-GPU, scheduling, recovery, and cost measurements into one benchmark schema.

Every unavailable backend is recorded as `not_run` with the detected reason. No missing hardware result is replaced with an estimate.

Run:

```bash
python benchmarks/final_system_benchmark.py
```

or run the complete Phase 11-15 study:

```bash
python scripts/run_distributed_study.py
```
