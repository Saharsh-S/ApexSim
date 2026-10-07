"""Triton implementation of the hot lap-time computation.

This module deliberately implements a *computational slice* of the Phase 3
CUDA race kernel rather than translating the whole race state machine. Laps
remain sequential at the race level; the hot arithmetic for all drivers in a
batch is evaluated as a dense tensor operation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # Triton is optional because CPU-only environments cannot import it.
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent.
    triton = None
    tl = None
    TRITON_AVAILABLE = False


@dataclass(frozen=True)
class TritonLapTimeResult:
    """Result of the Triton lap-time kernel."""

    lap_times: np.ndarray
    kernel_seconds: float
    block_size: int


def triton_available() -> bool:
    """Return whether Triton can be imported."""
    return TRITON_AVAILABLE


def triton_backend_status() -> str:
    """Return a human-readable Triton backend status."""
    if not TRITON_AVAILABLE:
        return "Triton is not installed"
    try:
        import torch

        if not torch.cuda.is_available():
            return "Triton installed, but no CUDA-capable PyTorch device is visible"
    except Exception as exc:
        return f"Triton installed; CUDA device check unavailable: {exc}"
    return "Triton and a CUDA-capable PyTorch device are available"


if TRITON_AVAILABLE:

    @triton.jit
    def _lap_time_kernel(
        total_ptr,
        fuel_ptr,
        tire_age_ptr,
        tire_id_ptr,
        ahead_index_ptr,
        lap_out_ptr,
        driver_skill_ptr,
        driver_aggression_ptr,
        driver_management_ptr,
        driver_consistency_ptr,
        car_performance_ptr,
        car_tire_performance_ptr,
        tire_pace_ptr,
        tire_degradation_ptr,
        tire_warmup_ptr,
        weather_mismatch_ptr,
        N_RACES: tl.constexpr,
        N_DRIVERS: tl.constexpr,
        BASE_LAP_TIME: tl.constexpr,
        FUEL_EFFECT: tl.constexpr,
        TRAFFIC_BASE: tl.constexpr,
        OVERTAKING_DIFFICULTY: tl.constexpr,
        DEGRADATION_MULTIPLIER: tl.constexpr,
        WEATHER_ID: tl.constexpr,
        WEATHER_BASE: tl.constexpr,
        NOISE_STD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute one lap's arithmetic for one race per Triton program.

        Grid: one program per race. Lanes represent drivers. BLOCK_SIZE is a
        power-of-two >= driver count; masked lanes are inactive. Driver and
        tire configuration is read as contiguous vectors, while per-race state
        is row-major [race, driver]. The result is one dense [race, driver]
        lap-time matrix.
        """
        race_id = tl.program_id(0)
        lanes = tl.arange(0, BLOCK_SIZE)
        mask = lanes < N_DRIVERS
        row = race_id * N_DRIVERS + lanes

        skill = tl.load(driver_skill_ptr + lanes, mask=mask, other=0.0)
        aggression = tl.load(driver_aggression_ptr + lanes, mask=mask, other=0.0)
        management = tl.load(driver_management_ptr + lanes, mask=mask, other=0.0)
        consistency = tl.load(driver_consistency_ptr + lanes, mask=mask, other=0.0)
        car_perf = tl.load(car_performance_ptr + lanes, mask=mask, other=1.0)
        car_tire_perf = tl.load(car_tire_performance_ptr + lanes, mask=mask, other=1.0)

        fuel = tl.load(fuel_ptr + row, mask=mask, other=0.0)
        age_i = tl.load(tire_age_ptr + row, mask=mask, other=0)
        tire_id = tl.load(tire_id_ptr + row, mask=mask, other=0)
        tire_id = tl.maximum(tire_id, 0)
        tire_id = tl.minimum(tire_id, 4)

        tire_pace = tl.load(tire_pace_ptr + tire_id, mask=mask, other=0.0)
        tire_deg = tl.load(tire_degradation_ptr + tire_id, mask=mask, other=0.0)
        tire_warm = tl.load(tire_warmup_ptr + tire_id, mask=mask, other=0.0)
        mismatch_offset = WEATHER_ID * 5 + tire_id
        mismatch = tl.load(weather_mismatch_ptr + mismatch_offset, mask=mask, other=0.0)

        age = age_i.to(tl.float32)
        management_factor = 1.0 - 0.35 * tl.maximum(0.0, tl.minimum(1.0, management))
        degradation = tire_deg * tl.pow(age, 1.35) * management_factor * DEGRADATION_MULTIPLIER
        warmup = tl.where(
            (tire_warm > 0.0) & (age < tire_warm),
            (tire_warm - age) / tire_warm * 0.8,
            0.0,
        )

        lap = (
            BASE_LAP_TIME
            + (0.5 - skill) * 2.4
            + (1.0 - car_perf) * 1.8
            - tire_pace * car_tire_perf
            + degradation
            + warmup
            + fuel * FUEL_EFFECT
            + WEATHER_BASE
            + mismatch
        )

        ahead = tl.load(ahead_index_ptr + row, mask=mask, other=-1)
        valid_ahead = mask & (ahead >= 0)
        ahead_row = race_id * N_DRIVERS + tl.maximum(ahead, 0)
        ahead_total = tl.load(total_ptr + ahead_row, mask=valid_ahead, other=0.0)
        current_total = tl.load(total_ptr + row, mask=mask, other=0.0)
        gap = tl.maximum(0.0, current_total - ahead_total)
        traffic = TRAFFIC_BASE * tl.maximum(0.0, 1.0 - gap / 3.0) * OVERTAKING_DIFFICULTY
        traffic *= 1.0 - 0.35 * aggression
        traffic = tl.where(valid_ahead, traffic, 0.0)
        lap += traffic

        # The full CUDA race kernel applies Gaussian noise in its RNG path.
        # This hot-kernel slice intentionally receives the already-sampled
        # state and leaves stochastic sampling to the orchestration layer.
        _ = NOISE_STD + consistency * 0.0
        lap = tl.maximum(1.0, lap)
        tl.store(lap_out_ptr + row, lap, mask=mask)


def _next_power_of_two(value: int) -> int:
    block = 1
    while block < value:
        block *= 2
    return block


def compute_lap_times_cpu(
    *,
    total: np.ndarray,
    fuel: np.ndarray,
    tire_age: np.ndarray,
    tire_id: np.ndarray,
    ahead_index: np.ndarray,
    driver_skill: np.ndarray,
    driver_aggression: np.ndarray,
    driver_management: np.ndarray,
    driver_consistency: np.ndarray,
    car_performance: np.ndarray,
    car_tire_performance: np.ndarray,
    tire_pace: np.ndarray,
    tire_degradation: np.ndarray,
    tire_warmup: np.ndarray,
    weather_mismatch: np.ndarray,
    weather_id: int,
    weather_base: float,
    base_lap_time: float,
    fuel_effect: float,
    traffic_base: float,
    overtaking_difficulty: float,
    degradation_multiplier: float,
) -> np.ndarray:
    """Reference implementation of the hot lap-time arithmetic on CPU."""
    races, drivers = total.shape
    out = np.empty((races, drivers), dtype=np.float32)
    for r in range(races):
        for i in range(drivers):
            tid = int(tire_id[r, i])
            age = float(tire_age[r, i])
            management_factor = 1.0 - 0.35 * np.clip(driver_management[i], 0.0, 1.0)
            degradation = tire_degradation[tid] * (age ** 1.35) * management_factor * degradation_multiplier
            warm = 0.0
            if tire_warmup[tid] > 0.0 and age < tire_warmup[tid]:
                warm = (tire_warmup[tid] - age) / tire_warmup[tid] * 0.8
            lap = (
                base_lap_time + (0.5 - driver_skill[i]) * 2.4
                + (1.0 - car_performance[i]) * 1.8
                - tire_pace[tid] * car_tire_performance[i]
                + degradation + warm + fuel[r, i] * fuel_effect
                + weather_base + weather_mismatch[weather_id, tid]
            )
            ahead = int(ahead_index[r, i])
            if ahead >= 0:
                gap = max(0.0, float(total[r, i]) - float(total[r, ahead]))
                traffic = traffic_base * max(0.0, 1.0 - gap / 3.0) * overtaking_difficulty
                traffic *= 1.0 - 0.35 * driver_aggression[i]
                lap += traffic
            out[r, i] = max(1.0, lap)
    return out


def compute_lap_times_triton(
    *,
    total: np.ndarray,
    fuel: np.ndarray,
    tire_age: np.ndarray,
    tire_id: np.ndarray,
    ahead_index: np.ndarray,
    driver_skill: np.ndarray,
    driver_aggression: np.ndarray,
    driver_management: np.ndarray,
    driver_consistency: np.ndarray,
    car_performance: np.ndarray,
    car_tire_performance: np.ndarray,
    tire_pace: np.ndarray,
    tire_degradation: np.ndarray,
    tire_warmup: np.ndarray,
    weather_mismatch: np.ndarray,
    weather_id: int,
    weather_base: float,
    base_lap_time: float,
    fuel_effect: float,
    traffic_base: float,
    overtaking_difficulty: float,
    degradation_multiplier: float,
    noise_std: float = 0.0,
    block_size: int | None = None,
) -> TritonLapTimeResult:
    """Evaluate the Phase 1/3 lap-time arithmetic with Triton.

    Inputs are the per-race state immediately before lap-time calculation.
    They are intentionally explicit so CUDA and Triton can be compared on the
    same state without requiring either implementation to own the race loop.

    Triton currently requires a CUDA backend; this function raises a clear
    error rather than silently falling back to CPU.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if total.ndim != 2:
        raise ValueError("total must have shape [races, drivers]")
    races, drivers = total.shape
    if drivers <= 0 or races <= 0:
        raise ValueError("total must contain at least one race and driver")
    if drivers > 1024:
        raise ValueError("Triton lap-time kernel supports at most 1024 drivers")
    if weather_id not in (0, 1, 2):
        raise ValueError("weather_id must be 0, 1, or 2")

    arrays = {
        "fuel": fuel,
        "tire_age": tire_age,
        "tire_id": tire_id,
        "ahead_index": ahead_index,
    }
    for name, value in arrays.items():
        if value.shape != total.shape:
            raise ValueError(f"{name} must have shape {total.shape}")

    driver_arrays = {
        "driver_skill": driver_skill,
        "driver_aggression": driver_aggression,
        "driver_management": driver_management,
        "driver_consistency": driver_consistency,
        "car_performance": car_performance,
        "car_tire_performance": car_tire_performance,
    }
    for name, value in driver_arrays.items():
        if value.shape != (drivers,):
            raise ValueError(f"{name} must have shape ({drivers},)")

    if block_size is None:
        block_size = _next_power_of_two(drivers)
    if block_size < drivers or block_size > 1024:
        raise ValueError("block_size must be a power-of-two >= driver count and <= 1024")

    # Import torch only when Triton is actually requested. This keeps the
    # optional backend importable in the normal CPU-only project environment.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Triton requires a CUDA-capable PyTorch device")

    device = torch.device("cuda")
    def tensor(value, dtype):
        return torch.as_tensor(value, dtype=dtype, device=device).contiguous()

    total_t = tensor(total, torch.float32)
    fuel_t = tensor(fuel, torch.float32)
    age_t = tensor(tire_age, torch.int32)
    tire_id_t = tensor(tire_id, torch.int32)
    ahead_t = tensor(ahead_index, torch.int32)
    driver_tensors = [tensor(driver_arrays[k], torch.float32) for k in driver_arrays]
    tire_pace_t = tensor(tire_pace, torch.float32)
    tire_deg_t = tensor(tire_degradation, torch.float32)
    tire_warm_t = tensor(tire_warmup, torch.float32)
    mismatch_t = tensor(weather_mismatch, torch.float32)
    out_t = torch.empty_like(total_t)

    # CUDA events measure kernel time without counting Python dispatch overhead.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _lap_time_kernel[(races,)](
        total_t, fuel_t, age_t, tire_id_t, ahead_t, out_t,
        *driver_tensors, tire_pace_t, tire_deg_t, tire_warm_t, mismatch_t,
        N_RACES=races,
        N_DRIVERS=drivers,
        BASE_LAP_TIME=float(base_lap_time),
        FUEL_EFFECT=float(fuel_effect),
        TRAFFIC_BASE=float(traffic_base),
        OVERTAKING_DIFFICULTY=float(overtaking_difficulty),
        DEGRADATION_MULTIPLIER=float(degradation_multiplier),
        WEATHER_ID=int(weather_id),
        WEATHER_BASE=float(weather_base),
        NOISE_STD=float(noise_std),
        BLOCK_SIZE=block_size,
    )
    end.record()
    end.synchronize()
    kernel_seconds = float(start.elapsed_time(end)) / 1000.0
    return TritonLapTimeResult(out_t.cpu().numpy(), kernel_seconds, block_size)
