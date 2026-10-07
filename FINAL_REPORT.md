# F1 GPU Race Simulator — Final Report

## Scope

Phases 1–15 form a single reproducible ML Systems study around F1 strategy search.

### Phase 1 — CPU reference
A deterministic race simulator establishes the correctness baseline.

### Phase 2 — Monte Carlo
Independent simulations and strategy statistics turn one race simulation into a search workload.

### Phase 3 — CUDA
Independent simulations are mapped to CUDA threads with per-simulation random streams.

### Phase 4 — Performance engineering
Profiling identifies redundant ranking, unnecessary stochastic work, and repeated configuration transfers. Controlled CUDA-simulator ablations quantify those changes.

### Phase 5 — Triton
A vectorized Triton hot operation provides an alternative GPU programming model.

### Phase 6 — Strategy search
Large candidate pools combine strategy generation, Monte Carlo evaluation, and ranking.

### Phase 7 — ML surrogate
XGBoost and PyTorch models approximate expensive simulation outcomes.

### Phase 8 — Inference optimization
Batching and FP32/FP16/BF16 inference are benchmarked, with optional ONNX/TensorRT/Triton paths.

### Phase 9 — Real-time engine
A two-tier engine uses a surrogate to screen candidates and simulation to validate finalists. Warm decision latency is measured rather than claimed.

### Phase 10 — Reproducible study
A controlled benchmark schema, reports, and benchmark runner make the experiment repeatable.

### Phase 11 — Multi-GPU distributed simulation
Monte Carlo simulation ranges are partitioned across independent workers. The system computes speedup and scaling efficiency for 1, 2, 4, and 8 workers and uses the same interface for real CUDA devices or an explicitly labeled NumPy fallback.

### Phase 12 — Simulated AI compute cluster
A priority and memory-aware GPU scheduler allocates simulation and inference jobs, tracks state, and retries failed work.

### Phase 13 — Fault tolerance
Checkpoints store completed ranges, failed workers, and metadata. Deterministic per-index seeds make replay independent of worker placement.

### Phase 14 — Cost-aware infrastructure
Runtime can be converted to cost and energy only when a real GPU price/power profile is supplied. No provider pricing is assumed.

### Phase 15 — Final benchmark
A consolidated benchmark table reports measured systems and explicitly records unavailable systems.

## Current hardware result

The release-validation environment has no CUDA driver/device. Therefore physical CUDA, Triton, multi-GPU, GPU memory, utilization, and power values are not claimed. CPU and CUDA-simulator results remain useful for correctness and software-level analysis.

## Reproduction

```bash
python -m pip install -e .
pytest -q
python scripts/run_all_benchmarks.py
python scripts/run_distributed_study.py
```

For a physical GPU system, install the optional CUDA/Triton/ML dependencies described in `pyproject.toml` and rerun the same scripts.

## Release quality

The repository excludes Git metadata, Python caches, virtual environments, temporary datasets, and temporary profiling files. Source, tests, benchmarks, documentation, scripts, examples, and reproducibility outputs are included.
