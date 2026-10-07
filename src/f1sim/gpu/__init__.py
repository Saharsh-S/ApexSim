"""GPU backends for the F1 Monte Carlo simulator.

CUDA remains the full-race GPU reference. Triton provides a separate
implementation of the measured hot lap-time computation.
"""

from .cuda import (
    GPURaceBatch,
    GPUExecutionTiming,
    GPUDeviceContext,
    create_gpu_context,
    cuda_available,
    cuda_backend_status,
    simulate_strategy_gpu,
)

__all__ = [
    "GPURaceBatch",
    "GPUExecutionTiming",
    "GPUDeviceContext",
    "create_gpu_context",
    "cuda_available",
    "cuda_backend_status",
    "simulate_strategy_gpu",
]
