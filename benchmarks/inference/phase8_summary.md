# Phase 8 Benchmark Summary

Environment: CPU-only. No NVIDIA CUDA device, ONNX Runtime, TensorRT, or Triton was available.

| Implementation | Latency | Throughput | Accuracy |
|---|---:|---:|---|
| XGBoost CPU | 3.77 ms @ batch 1 | 265 samples/s | measured in Phase 7 evaluation |
| PyTorch FP32 CPU | 0.0245 ms @ batch 1 | 40.9k samples/s | reference learned model |
| PyTorch FP16 CPU | measured in `phase8_results.json` | measured | compared with FP32 |
| PyTorch BF16 CPU | measured in `phase8_results.json` | measured | compared with FP32 |
| ONNX Runtime | unavailable | unavailable | unavailable |
| TensorRT | unavailable | unavailable | unavailable |
| Triton | unavailable | unavailable | unavailable |

These are actual CPU measurements, not GPU estimates. GPU latency and memory require a CUDA-equipped environment.
