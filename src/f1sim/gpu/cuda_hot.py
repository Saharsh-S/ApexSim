"""CUDA implementation of the Phase 4 hot lap-time computation.

This is a benchmark/comparison kernel, not a replacement for the full CUDA
race engine in :mod:`f1sim.gpu.cuda`.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import cuda, float32, int32
    CUDA_HOT_AVAILABLE = True
except Exception:  # pragma: no cover
    cuda = None
    float32 = int32 = None
    CUDA_HOT_AVAILABLE = False


if CUDA_HOT_AVAILABLE:
    @cuda.jit
    def _lap_time_kernel(
        total, fuel, tire_age, tire_id, ahead_index, out,
        skill, aggression, management, car_perf, car_tire_perf,
        tire_pace, tire_deg, tire_warm, mismatch,
        base_lap_time, fuel_effect, traffic_base, overtaking_difficulty,
        degradation_multiplier, weather_id, weather_base,
    ):
        race = cuda.grid(1)
        if race >= total.shape[0]:
            return
        for i in range(total.shape[1]):
            tid = tire_id[race, i]
            age = float(tire_age[race, i])
            management_factor = 1.0 - 0.35 * max(0.0, min(1.0, management[i]))
            degradation = tire_deg[tid] * (age ** 1.35) * management_factor * degradation_multiplier
            warm = 0.0
            if tire_warm[tid] > 0.0 and age < tire_warm[tid]:
                warm = (tire_warm[tid] - age) / tire_warm[tid] * 0.8
            lap = (
                base_lap_time
                + (0.5 - skill[i]) * 2.4
                + (1.0 - car_perf[i]) * 1.8
                - tire_pace[tid] * car_tire_perf[i]
                + degradation + warm
                + fuel[race, i] * fuel_effect
                + weather_base + mismatch[weather_id, tid]
            )
            ahead = ahead_index[race, i]
            if ahead >= 0:
                gap = max(0.0, total[race, i] - total[race, ahead])
                traffic = traffic_base * max(0.0, 1.0 - gap / 3.0) * overtaking_difficulty
                traffic *= 1.0 - 0.35 * aggression[i]
                lap += traffic
            out[race, i] = max(1.0, lap)
else:
    _lap_time_kernel = None


def cuda_hot_available() -> bool:
    """Return whether Numba can see a real CUDA device."""
    if not CUDA_HOT_AVAILABLE:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def compute_lap_times_cuda(
    *, state: dict[str, np.ndarray], constants: dict[str, np.ndarray | float | int], threads_per_block: int = 128
) -> tuple[np.ndarray, float]:
    """Compute the hot lap-time slice with CUDA and return output + kernel seconds."""
    if not cuda_hot_available():
        raise RuntimeError("CUDA hot kernel requires a real CUDA device")
    total = np.ascontiguousarray(state["total"], dtype=np.float32)
    fuel = np.ascontiguousarray(state["fuel"], dtype=np.float32)
    tire_age = np.ascontiguousarray(state["tire_age"], dtype=np.int32)
    tire_id = np.ascontiguousarray(state["tire_id"], dtype=np.int32)
    ahead_index = np.ascontiguousarray(state["ahead_index"], dtype=np.int32)
    d = total.shape[1]
    output = cuda.device_array_like(total)
    arrays = [cuda.to_device(x) for x in (total, fuel, tire_age, tire_id, ahead_index)]
    c = constants
    cfg = [cuda.to_device(np.asarray(c[k], dtype=np.float32)) for k in (
        "skill", "aggression", "management", "car_perf", "car_tire_perf",
        "tire_pace", "tire_deg", "tire_warm", "mismatch")]
    start = cuda.event()
    end = cuda.event()
    start.record()
    blocks = (total.shape[0] + threads_per_block - 1) // threads_per_block
    _lap_time_kernel[blocks, threads_per_block](*arrays, output, *cfg,
        float(c["base_lap_time"]), float(c["fuel_effect"]), float(c["traffic_base"]),
        float(c["overtaking_difficulty"]), float(c["degradation_multiplier"]),
        int(c["weather_id"]), float(c["weather_base"]))
    end.record()
    end.synchronize()
    seconds = float(cuda.event_elapsed_time(start, end)) / 1000.0
    return output.copy_to_host(), seconds
