"""CUDA Monte Carlo engine using Numba CUDA.

The GPU path maps one CUDA thread to one independent race.  The CPU simulator
remains the correctness reference; this module mirrors its race equations and
state transitions while using a device-local RNG stream per simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..config import SimulationConfig, TireCompound, WeatherType
from ..models import Strategy
from ..strategy import validate_strategy

try:  # Importing numba itself is safe without a CUDA driver.
    from numba import cuda, float64, int32, int64, uint64
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment.
    cuda = None
    _NUMBA_AVAILABLE = False

MAX_GPU_DRIVERS = 16


@dataclass(frozen=True)
class GPUExecutionTiming:
    """Host-side timing breakdown for one GPU batch.

    Times are measured with ``perf_counter`` around the corresponding CUDA
    operations. On a real GPU, kernel-only timing should be cross-checked with
    CUDA events or Nsight because asynchronous launch semantics can hide work.
    """

    input_encoding_seconds: float
    host_to_device_seconds: float
    kernel_seconds: float
    device_to_host_seconds: float
    total_seconds: float
    host_to_device_bytes: int
    device_to_host_bytes: int
    sort_comparisons: int = 0
    estimated_rng_draws: int = 0


@dataclass(frozen=True)
class GPUDeviceContext:
    """Persistent immutable configuration uploaded once to a CUDA device."""

    config: SimulationConfig
    driver_count: int
    skill: object
    consistency: object
    aggression: object
    management: object
    car_performance: object
    car_reliability: object
    car_tire_performance: object
    tire_pace: object
    tire_degradation: object
    tire_warmup: object
    weather_mismatch: object


@dataclass(frozen=True)
class GPURaceBatch:
    """Compact GPU output buffers copied back to the host."""

    seeds: np.ndarray
    race_times: np.ndarray
    finishing_positions: np.ndarray
    pit_stops: np.ndarray
    dnf: np.ndarray
    fastest_laps: np.ndarray
    safety_car_laps: np.ndarray
    simulations: int
    strategy_id: str
    timing: GPUExecutionTiming | None = None

    @property
    def simulations_per_second_fields(self) -> int:
        """Return the number of simulations represented by this batch."""
        return self.simulations


def cuda_available() -> bool:
    """Return whether a real CUDA device is visible to Numba."""
    if not _NUMBA_AVAILABLE:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def cuda_backend_status() -> str:
    """Return a human-readable CUDA availability status."""
    if not _NUMBA_AVAILABLE:
        return "Numba is not installed"
    try:
        if cuda.is_available():
            return "CUDA device available"
    except Exception as exc:
        return f"CUDA unavailable: {exc}"
    return "CUDA driver/device unavailable"


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def _strategy_id(strategy: Strategy) -> str:
    compounds = [strategy.initial_compound.value, *[c.value for c in strategy.compounds]]
    if not strategy.pit_laps:
        return "-".join(compounds)
    return "-".join(f"{compound}@{lap}" for compound, lap in zip(compounds, [0, *strategy.pit_laps]))


if _NUMBA_AVAILABLE:
    @cuda.jit(device=True, inline=True)
    def _rng_next(state):
        """Independent per-thread 64-bit LCG stream."""
        state[0] = state[0] * uint64(6364136223846793005) + uint64(1442695040888963407)
        return (state[0] >> uint64(11)) * (1.0 / 9007199254740992.0)


    @cuda.jit(device=True, inline=True)
    def _normal(state):
        u1 = _rng_next(state)
        u2 = _rng_next(state)
        if u1 < 1e-12:
            u1 = 1e-12
        # Avoid importing a host-side RNG implementation into the kernel.
        import math
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    @cuda.jit
    def _race_kernel(
        seeds,
        race_times,
        positions_out,
        pit_stops_out,
        dnf_out,
        fastest_out,
        safety_out,
        initial_tire_ids,
        pit_laps,
        pit_tire_ids,
        driver_skill,
        driver_consistency,
        driver_aggression,
        driver_management,
        car_performance,
        car_reliability,
        car_tire_performance,
        tire_pace,
        tire_degradation,
        tire_warmup,
        weather_mismatch,
        track_laps,
        base_lap_time,
        pit_loss,
        overtaking_difficulty,
        degradation_multiplier,
        fuel_effect,
        traffic_base,
        noise_std,
        dnf_base,
        weather_initial,
        weather_transition_probability,
        dry_penalty,
        light_rain_penalty,
        heavy_rain_penalty,
        safety_probability,
        safety_duration,
        safety_gap_compression,
        driver_count,
        profiling_counts,
        profile_enabled,
    ):
        """One independent complete race per CUDA thread."""
        race = cuda.grid(1)
        if race >= seeds.size:
            return

        import math
        state = cuda.local.array(1, dtype=uint64)
        # Mix the batch seed so adjacent simulation IDs cannot share a short prefix.
        state[0] = uint64(seeds[race]) * uint64(2862933555777941757) + uint64(3037000493)
        state[0] ^= state[0] >> uint64(33)
        state[0] *= uint64(0xFF51AFD7ED558CCD)
        state[0] ^= state[0] >> uint64(33)

        total = cuda.local.array(MAX_GPU_DRIVERS, dtype=float64)
        fuel = cuda.local.array(MAX_GPU_DRIVERS, dtype=float64)
        tire_age = cuda.local.array(MAX_GPU_DRIVERS, dtype=int32)
        tire_id = cuda.local.array(MAX_GPU_DRIVERS, dtype=int32)
        dnf = cuda.local.array(MAX_GPU_DRIVERS, dtype=int32)
        stops = cuda.local.array(MAX_GPU_DRIVERS, dtype=int32)
        fastest = cuda.local.array(MAX_GPU_DRIVERS, dtype=float64)
        order = cuda.local.array(MAX_GPU_DRIVERS, dtype=int32)
        pit_index = cuda.local.array(MAX_GPU_DRIVERS, dtype=int32)
        position = cuda.local.array(MAX_GPU_DRIVERS, dtype=int32)

        for i in range(MAX_GPU_DRIVERS):
            total[i] = 0.0
            fuel[i] = 1.0
            tire_age[i] = 0
            tire_id[i] = initial_tire_ids[i] if i < driver_count else 0
            dnf[i] = 0
            stops[i] = 0
            fastest[i] = 1e30
            order[i] = i if i < driver_count else 0
            pit_index[i] = 0
            position[i] = i + 1 if i < driver_count else 0

        weather = weather_initial
        safety_active = 0
        safety_remaining = 0
        safety_laps = 0

        for lap in range(1, track_laps + 1):
            # Phase 1 weather transition: transition draw + state-dependent draw.
            if weather_transition_probability > 0.0 and _rng_next(state) < weather_transition_probability:
                draw = _rng_next(state)
                if weather == 0:
                    weather = 1 if draw < 0.75 else 2
                elif weather == 1:
                    weather = 0 if draw < 0.55 else 2
                else:
                    weather = 1 if draw < 0.65 else 0

            if safety_active == 0 and safety_probability > 0.0 and _rng_next(state) < safety_probability:
                safety_active = 1
                safety_remaining = safety_duration

            # Pit stops occur before lap-time calculation in Phase 1. A
            # post-lap ranking is already current when there was no pit stop,
            # so only re-rank when a pit changes a driver's accumulated time.
            needs_rank = 0
            for i in range(driver_count):
                if dnf[i] == 0 and pit_index[i] < pit_laps.shape[0]:
                    if pit_laps[pit_index[i], i] == lap:
                        total[i] += pit_loss
                        stops[i] += 1
                        tire_id[i] = pit_tire_ids[pit_index[i], i]
                        tire_age[i] = 0
                        pit_index[i] += 1
                        needs_rank = 1

            if needs_rank != 0:
                for i in range(driver_count):
                    order[i] = i
                for a in range(driver_count):
                    best = a
                    for b in range(a + 1, driver_count):
                        ib = order[b]
                        ia = order[best]
                        if total[ib] < total[ia] or (total[ib] == total[ia] and ib < ia):
                            best = b
                    temp = order[a]
                    order[a] = order[best]
                    order[best] = temp
                for p in range(driver_count):
                    position[order[p]] = p + 1

            for i in range(driver_count):
                if dnf[i] != 0:
                    continue
                reliability_probability = dnf_base + (1.0 - car_reliability[i]) * 0.012
                if reliability_probability > 0.0 and _rng_next(state) < reliability_probability:
                    dnf[i] = 1
                    continue

                tid = tire_id[i]
                age = float(tire_age[i])
                management_factor = 1.0 - 0.35 * max(0.0, min(1.0, driver_management[i]))
                degradation = tire_degradation[tid] * (age ** 1.35) * management_factor * degradation_multiplier
                warm = 0.0
                if tire_warmup[tid] > 0.0 and age < tire_warmup[tid]:
                    warm = (tire_warmup[tid] - age) / tire_warmup[tid] * 0.8

                lap_time = base_lap_time
                lap_time += (0.5 - driver_skill[i]) * 2.4
                lap_time += (1.0 - car_performance[i]) * 1.8
                lap_time -= tire_pace[tid] * car_tire_performance[i]
                lap_time += degradation + warm
                lap_time += fuel[i] * fuel_effect
                if weather == 0:
                    lap_time += dry_penalty + weather_mismatch[0, tid]
                elif weather == 1:
                    lap_time += light_rain_penalty + weather_mismatch[1, tid]
                else:
                    lap_time += heavy_rain_penalty + weather_mismatch[2, tid]

                p = position[i]
                gap_ahead = 10.0
                if p > 1:
                    ahead = order[p - 2]
                    gap_ahead = max(0.0, total[i] - total[ahead])
                traffic = traffic_base * max(0.0, 1.0 - gap_ahead / 3.0) * overtaking_difficulty
                traffic *= (1.0 - 0.35 * driver_aggression[i])
                lap_time += traffic

                noise_scale = noise_std * (1.2 - driver_consistency[i])
                if noise_scale != 0.0:
                    lap_time += _normal(state) * noise_scale
                if lap_time < 1.0:
                    lap_time = 1.0
                if safety_active != 0:
                    lap_time = 0.88 * lap_time + 0.12 * base_lap_time

                total[i] += lap_time
                if lap_time < fastest[i]:
                    fastest[i] = lap_time
                tire_age[i] += 1
                fuel[i] = max(0.0, fuel[i] - 1.0 / track_laps)

            # Re-rank after the lap and update safety-car state.
            for i in range(driver_count):
                order[i] = i
            for a in range(driver_count):
                best = a
                for b in range(a + 1, driver_count):
                    if profile_enabled != 0:
                        profiling_counts[race, 0] += 1
                    ib = order[b]
                    ia = order[best]
                    tb = 1e30 if dnf[ib] else total[ib]
                    ta = 1e30 if dnf[ia] else total[ia]
                    if tb < ta or (tb == ta and ib < ia):
                        best = b
                temp = order[a]
                order[a] = order[best]
                order[best] = temp
            for p in range(driver_count):
                position[order[p]] = p + 1

            if safety_active != 0:
                safety_laps += 1
                safety_remaining -= 1
                if safety_remaining <= 0:
                    safety_active = 0

        race_time = 0.0
        any_dnf = 0
        for i in range(driver_count):
            if dnf[i] != 0:
                any_dnf = 1
            elif total[i] > race_time:
                race_time = total[i]
        for i in range(driver_count):
            if dnf[i] != 0:
                position[i] = -1
        race_times[race] = race_time
        safety_out[race] = safety_laps
        dnf_out[race] = any_dnf
        for i in range(driver_count):
            positions_out[race, i] = position[i]
            pit_stops_out[race, i] = stops[i]
            fastest_out[race, i] = fastest[i] if fastest[i] < 1e29 else -1.0

else:
    _race_kernel = None


def _encode_inputs(config: SimulationConfig, strategy: Strategy, simulations: int, seed: int):
    validate_strategy(strategy, config)
    driver_count = len(config.drivers)
    if driver_count > MAX_GPU_DRIVERS:
        raise ValueError(f"GPU engine supports at most {MAX_GPU_DRIVERS} drivers per race")

    # GPU semantics match Phase 2 simulate_strategy: focal strategy for driver 0,
    # Phase 1's fixed Medium->Hard baseline for all other drivers.
    baseline_lap = max(1, min(config.track.number_of_laps - 1, config.track.number_of_laps // 2))
    baseline = Strategy(TireCompound.MEDIUM, [baseline_lap], [TireCompound.HARD])
    strategies = [strategy] + [baseline] * (driver_count - 1)
    compounds = list(TireCompound)
    compound_index = {c: i for i, c in enumerate(compounds)}

    max_stops = max(s.number_of_stops for s in strategies)
    pit_laps = np.full((max_stops, driver_count), -1, dtype=np.int32)
    pit_tires = np.zeros((max_stops, driver_count), dtype=np.int32)
    initial_tires = np.zeros(driver_count, dtype=np.int32)
    for i, s in enumerate(strategies):
        initial_tires[i] = compound_index[s.initial_compound]
        for j, lap in enumerate(s.pit_laps):
            pit_laps[j, i] = lap
            pit_tires[j, i] = compound_index[s.compounds[j]]

    weather_mismatch = np.array([
        [0.0, 0.0, 0.0, 9.0, 15.0],
        [4.0, 3.0, 3.5, 0.0, 2.0],
        [11.0, 10.0, 10.0, 2.0, 0.0],
    ], dtype=np.float64)

    def arr(values, dtype=np.float64):
        return np.asarray(values, dtype=dtype)

    return (
        np.arange(seed, seed + simulations, dtype=np.int64),
        initial_tires,
        pit_laps,
        pit_tires,
        arr([d.skill_rating for d in config.drivers]),
        arr([d.consistency for d in config.drivers]),
        arr([d.aggression for d in config.drivers]),
        arr([d.tire_management for d in config.drivers]),
        arr([c.performance_coefficient for c in config.cars]),
        arr([c.reliability for c in config.cars]),
        arr([c.tire_performance_coefficient for c in config.cars]),
        arr([config.tires[c].pace_delta for c in compounds]),
        arr([config.tires[c].degradation_rate for c in compounds]),
        arr([config.tires[c].warmup_laps for c in compounds]),
        weather_mismatch,
    )


def create_gpu_context(config: SimulationConfig) -> GPUDeviceContext:
    """Upload immutable simulation configuration once for repeated GPU batches."""
    if not cuda_available():
        raise RuntimeError(f"CUDA backend unavailable: {cuda_backend_status()}")
    encoded = _encode_inputs(config, Strategy(TireCompound.MEDIUM, [], []), 1, 0)
    (_, _, _, _, skill, consistency, aggression, management, car_perf,
     reliability, car_tire_perf, tire_pace, tire_deg, tire_warm, mismatch) = encoded
    return GPUDeviceContext(
        config=config,
        driver_count=len(config.drivers),
        skill=cuda.to_device(skill),
        consistency=cuda.to_device(consistency),
        aggression=cuda.to_device(aggression),
        management=cuda.to_device(management),
        car_performance=cuda.to_device(car_perf),
        car_reliability=cuda.to_device(reliability),
        car_tire_performance=cuda.to_device(car_tire_perf),
        tire_pace=cuda.to_device(tire_pace),
        tire_degradation=cuda.to_device(tire_deg),
        tire_warmup=cuda.to_device(tire_warm),
        weather_mismatch=cuda.to_device(mismatch),
    )


def simulate_strategy_gpu(
    config: SimulationConfig,
    strategy: Strategy,
    simulations: int,
    seed: int = 0,
    threads_per_block: int = 128,
    return_timing: bool = False,
    context: GPUDeviceContext | None = None,
) -> GPURaceBatch:
    """Run independent races on a CUDA device, one thread per race.

    Host-to-device transfer consists of small immutable configuration arrays and
    the seed array. Device-to-host transfer consists only of compact output
    metrics. Per-race mutable state remains thread-local on the GPU.
    """
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be positive")
    if not cuda_available():
        raise RuntimeError(f"CUDA backend unavailable: {cuda_backend_status()}")

    overall_start = perf_counter()
    encoded = _encode_inputs(config, strategy, simulations, seed)
    encode_done = perf_counter()
    (seeds, initial, pit_laps, pit_tires, skill, consistency, aggression,
     management, car_perf, reliability, car_tire_perf, tire_pace,
     tire_deg, tire_warm, mismatch) = encoded
    d = len(config.drivers)
    if context is not None and context.config != config:
        raise ValueError("GPU context was created for a different SimulationConfig")

    # Strategy/seed data changes per batch; immutable driver/tire configuration
    # can optionally stay resident on the device across many strategy launches.
    transfer_start = perf_counter()
    d_seeds = cuda.to_device(seeds)
    d_initial = cuda.to_device(initial)
    d_pit_laps = cuda.to_device(pit_laps)
    d_pit_tires = cuda.to_device(pit_tires)
    if context is None:
        device_arrays = [cuda.to_device(x) for x in (
            skill, consistency, aggression, management, car_perf,
            reliability, car_tire_perf, tire_pace, tire_deg, tire_warm, mismatch,
        )]
        static_bytes = sum(x.nbytes for x in (skill, consistency, aggression, management, car_perf, reliability, car_tire_perf, tire_pace, tire_deg, tire_warm, mismatch))
    else:
        device_arrays = [context.skill, context.consistency, context.aggression, context.management,
                         context.car_performance, context.car_reliability, context.car_tire_performance,
                         context.tire_pace, context.tire_degradation, context.tire_warmup, context.weather_mismatch]
        static_bytes = 0
    transfer_done = perf_counter()
    race_times = cuda.device_array(simulations, dtype=np.float64)
    positions = cuda.device_array((simulations, d), dtype=np.int32)
    stops = cuda.device_array((simulations, d), dtype=np.int32)
    dnfs = cuda.device_array(simulations, dtype=np.int32)
    fastest = cuda.device_array((simulations, d), dtype=np.float64)
    safety = cuda.device_array(simulations, dtype=np.int32)
    profile_enabled = 1 if return_timing else 0
    profile_shape = (simulations, 1) if return_timing else (1, 1)
    profiling_counts = cuda.to_device(np.zeros(profile_shape, dtype=np.int64))

    blocks = (simulations + threads_per_block - 1) // threads_per_block
    kernel_start = perf_counter()
    _race_kernel[blocks, threads_per_block](
        d_seeds, race_times, positions, stops, dnfs, fastest, safety,
        d_initial, d_pit_laps, d_pit_tires, *device_arrays,
        config.track.number_of_laps, config.track.base_lap_time,
        config.track.pit_lane_time_loss, config.track.overtaking_difficulty,
        config.track.tire_degradation_multiplier, config.fuel_effect_per_lap,
        config.traffic_base_penalty, config.stochastic_noise_std,
        config.dnf_base_probability, 0 if config.weather.initial == WeatherType.DRY else (1 if config.weather.initial == WeatherType.LIGHT_RAIN else 2),
        config.weather.transition_probability, config.weather.dry_penalty,
        config.weather.light_rain_penalty, config.weather.heavy_rain_penalty,
        config.weather.safety_car_probability, config.weather.safety_car_duration_laps,
        config.safety_car_gap_compression, d, profiling_counts, profile_enabled,
    )
    cuda.synchronize()
    kernel_done = perf_counter()

    d2h_start = perf_counter()
    host_race_times = race_times.copy_to_host()
    host_positions = positions.copy_to_host()
    host_stops = stops.copy_to_host()
    host_dnfs = dnfs.copy_to_host().astype(bool)
    host_fastest = fastest.copy_to_host()
    host_safety = safety.copy_to_host()
    host_profile = profiling_counts.copy_to_host() if return_timing else None
    d2h_done = perf_counter()

    estimated_rng_draws = simulations * config.track.number_of_laps * (2 + 3 * d)
    timing = GPUExecutionTiming(
        input_encoding_seconds=encode_done - overall_start,
        host_to_device_seconds=transfer_done - transfer_start,
        kernel_seconds=kernel_done - kernel_start,
        device_to_host_seconds=d2h_done - d2h_start,
        total_seconds=d2h_done - overall_start,
        host_to_device_bytes=int(seeds.nbytes + initial.nbytes + pit_laps.nbytes + pit_tires.nbytes + static_bytes),
        device_to_host_bytes=int(host_race_times.nbytes + host_positions.nbytes + host_stops.nbytes + host_dnfs.nbytes + host_fastest.nbytes + host_safety.nbytes),
        sort_comparisons=int(host_profile[:, 0].sum()) if host_profile is not None else 0,
        estimated_rng_draws=int(estimated_rng_draws),
    ) if return_timing else None


    return GPURaceBatch(
        seeds=seeds.copy(),
        race_times=host_race_times,
        finishing_positions=host_positions,
        pit_stops=host_stops,
        dnf=host_dnfs,
        fastest_laps=host_fastest,
        safety_car_laps=host_safety,
        simulations=simulations,
        strategy_id=_strategy_id(strategy),
        timing=timing,
    )
