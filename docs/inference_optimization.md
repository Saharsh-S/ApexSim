# Phase 8: Low-Latency ML Inference

Phase 8 evaluates the Phase 7 strategy-outcome model as an inference system rather than a training system.

## Backends

- PyTorch: reference learned model.
- ONNX Runtime: optional graph-runtime path when `onnx` and `onnxruntime` are installed.
- TensorRT: optional NVIDIA deployment path; no numbers are recorded unless TensorRT and a CUDA device are available.
- Triton: optional kernel path for targeted GPU work; it does not replace the CUDA simulator or PyTorch model.
- XGBoost CPU: included as a practical low-latency baseline.

## Batch sizes

`1, 8, 32, 128, 512, 1024, 4096`.

Latency is reported separately from throughput. GPU results should use CUDA events for device-only latency and a second timing mode for end-to-end latency including host/device transfers.

## Precision

FP32 is the reference. FP16 and BF16 are only enabled when the backend/device supports them. INT8 requires calibration and is not silently enabled: the project must demonstrate that its validation metrics remain within the documented tolerance before accepting an INT8 result.

## Accuracy tolerance

The learned model is compared against the held-out Monte Carlo labels. Phase 8 does not accept a faster backend as an improvement if its prediction error materially exceeds the PyTorch FP32 reference. The default acceptance criterion is no more than 2% relative increase in race-time MAE and no more than 0.02 absolute increase in win/podium MAE.

## Current environment limitation

The development environment used for this phase does not expose an NVIDIA CUDA device and does not provide ONNX Runtime, TensorRT, or Triton. Therefore CUDA/TensorRT/Triton performance numbers are not fabricated. The benchmark records them as unavailable and remains runnable on a CUDA-equipped machine.

## Tradeoff

The goal is not to declare one backend universally best. Batch size, transfer overhead, precision, graph compilation, and GPU availability can reverse the ranking. Single-sample inference is often dominated by launch/transfer overhead, while large batches amortize those costs.

## Triton kernel scope

The optional `f1sim.ml.triton_mlp.fused_bias_relu` kernel targets a concrete inference bottleneck: bias addition followed by ReLU on an intermediate activation. It is deliberately a micro-kernel experiment rather than a second full model runtime. A production Triton backend should fuse the complete MLP only after profiling shows that framework launch/intermediate-memory overhead dominates.
