"""Optional Triton implementation of the hot lap-time computation."""

from .lap_time import (
    TRITON_AVAILABLE,
    TritonLapTimeResult,
    triton_available,
    triton_backend_status,
    compute_lap_times_triton,
    compute_lap_times_cpu,
)

__all__ = [
    "TRITON_AVAILABLE",
    "TritonLapTimeResult",
    "triton_available",
    "triton_backend_status",
    "compute_lap_times_triton",
    "compute_lap_times_cpu",
]
