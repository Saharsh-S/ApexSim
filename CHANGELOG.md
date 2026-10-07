# Changelog

## Phase 10 - Reproducible ML Systems Performance Study

- Converted the project into a reproducible end-to-end performance study.
- Added `benchmarks/performance_study.py` for controlled simulation, inference, accuracy, and real-time measurements.
- Added `scripts/run_all_benchmarks.py` as the single benchmark entry point.
- Added machine-readable results under `results/`.
- Added generated throughput, latency, batching, and optimization graphs.
- Added explicit `not_run` records for unavailable CUDA/Triton/TensorRT/GPU/power measurements.
- Added `docs/performance_report.md` with methodology, results, optimization analysis, limitations, and future work.
- Added `FINAL_REPORT.md` summarizing Phases 1-10.
- Rewrote the README as a publication-ready open-source project description.
- Preserved historical CUDA-simulator measurements while explicitly separating them from real GPU results.

## Phase 9 - Real-Time Strategy Engine

- Added `RaceState` for live-like race-state input.
- Added `RealtimeStrategyEngine` for continuous candidate generation, evaluation, ranking, and decision output.
- Added weather-aware Intermediate/Wet candidate generation.
- Added surrogate and Monte Carlo evaluation modes.
- Added stage-level decision latency instrumentation.
- Added `examples/realtime_strategy.py`.
- Added `docs/realtime_architecture.md`.
- Added real-time engine regression tests.

## Phase 8 - Low-Latency ML Inference

- Added low-latency inference benchmarking for the Phase 7 model.
- Added PyTorch batch inference measurements across batch sizes 1, 8, 32, 128, 512, 1024, and 4096.
- Added optional ONNX export, TensorRT availability detection, and Triton availability detection.
- Added inference optimization documentation covering precision, batching, transfers, fusion, and accuracy tolerances.

## Phase 7 - ML Strategy Surrogate

- Added synthetic strategy-state/outcome dataset generation.
- Added XGBoost baseline models for race-time and finishing outcomes.
- Added a small PyTorch multi-task MLP.
- Added persisted model checkpoints and ML regression tests.
- Added ML evaluation infrastructure for strategy search approximation.

## Phase 6 - GPU Strategy Search

- Added procedural generation of large valid strategy candidate sets.
- Added CPU Python/NumPy and CUDA-backed strategy search.
- Added strategy ranking, visualization, and benchmark infrastructure.

## Phase 5 - Triton Hot-Kernel Experiment

- Added optional Triton backend for the selected lap-time arithmetic hot path.
- Added CUDA-vs-Triton comparison infrastructure and documentation.

## Phase 4 - CUDA Performance Engineering

- Added GPU timing instrumentation, persistent device context, conditional stochastic work, ranking-work instrumentation, and profiling workflows.
- Recorded CUDA-simulator optimization measurements without treating them as real GPU performance.

## Phase 3 - CUDA GPU Acceleration

- Added Numba CUDA Monte Carlo execution with deterministic per-simulation streams.
- Added CPU-reference and GPU determinism validation.

## Phase 2 - Monte Carlo

- Added Python and NumPy Monte Carlo execution, strategy generation, statistics, ranking, convergence analysis, and benchmarks.

## Phase 1 - CPU Reference Simulator

- Added the deterministic modular F1 race simulator with driver, car, tire, weather, fuel, traffic, safety-car, DNF, pit-stop, and seeded stochastic models.

## Phase 11–15 — Distributed AI Systems

- Added multi-worker Monte Carlo partitioning with CUDA backend and explicit NumPy fallback.
- Added speedup/scaling-efficiency measurements for 1/2/4/8 workers.
- Added GPU-aware priority scheduler with memory constraints and retries.
- Added checkpoint/recovery primitives and deterministic per-simulation seed derivation.
- Added transparent cost/energy model that requires real supplied pricing/power data.
- Added consolidated final AI systems benchmark and Phase 11–15 reproducibility runner.
- Added distributed scaling and final benchmark graphs.
- Added final systems documentation and release report.
