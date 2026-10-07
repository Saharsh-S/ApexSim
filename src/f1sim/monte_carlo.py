"""Monte Carlo race simulation engines."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .config import SimulationConfig, TireCompound, WeatherType
from .models import DriverResult, RaceResult, Strategy
from .simulator import simulate_race
from .tires import tire_degradation_penalty, tire_warmup_penalty
from .weather import weather_penalty
from .strategy import validate_strategy


@dataclass(frozen=True)
class MonteCarloRace:
    """Compact per-race outcome used by the Monte Carlo engine."""

    seed: int
    winner: str | None
    finishing_position: tuple[int | None, ...]
    total_race_time: float
    fastest_lap: float | None
    pit_stops: int
    tire_usage: tuple[str, ...]
    safety_car_laps: int
    dnf: bool
    total_fuel_used: float
    strategy_id: str
    race_result: RaceResult | None = None


@dataclass(frozen=True)
class StrategyStatistics:
    """Aggregate statistics for one strategy."""

    strategy_id: str
    simulations: int
    mean_race_time: float
    median_race_time: float
    std_race_time: float
    percentile_5: float
    percentile_95: float
    win_probability: float
    podium_probability: float
    dnf_probability: float
    mean_pit_stops: float
    mean_safety_car_laps: float
    mean_fuel_used: float
    risk_adjusted_score: float


@dataclass(frozen=True)
class MonteCarloResult:
    """All outcomes and aggregate statistics for a strategy."""

    strategy: Strategy
    strategy_id: str
    seeds: tuple[int, ...]
    races: tuple[MonteCarloRace, ...]
    statistics: StrategyStatistics


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing result for a Monte Carlo engine."""

    strategy_count: int
    simulations_per_strategy: int
    total_simulations: int
    runtime_seconds: float
    races_per_second: float
    peak_memory_mb: float | None = None


def strategy_id(strategy: Strategy) -> str:
    """Return a stable human-readable strategy identifier."""
    compounds = [strategy.initial_compound.value, *[c.value for c in strategy.compounds]]
    if not strategy.pit_laps:
        return "-".join(compounds)
    return "-".join(f"{compound}@{lap}" for compound, lap in zip(compounds, [0, *strategy.pit_laps]))


def _extract_race(result: RaceResult, config: SimulationConfig, sid: str) -> MonteCarloRace:
    classified = [r for r in result.driver_results if not r.dnf]
    winner = classified[0].driver if classified else None
    position_by_name = {r.driver: r.finishing_position for r in result.driver_results}
    positions = tuple(position_by_name.get(driver.name) for driver in config.drivers)
    lap_candidates = [r.fastest_lap for r in result.driver_results if r.fastest_lap is not None]
    fastest = min(lap_candidates) if lap_candidates else None
    stops = sum(r.pit_stops for r in result.driver_results)
    usage = tuple(sorted({compound.value for r in result.driver_results for compound in r.tire_usage}))
    dnf = any(r.dnf for r in result.driver_results)
    fuel_used = float(len(config.drivers))
    return MonteCarloRace(
        seed=result.seed,
        winner=winner,
        finishing_position=positions,
        total_race_time=result.total_race_time,
        fastest_lap=fastest,
        pit_stops=stops,
        tire_usage=usage,
        safety_car_laps=result.safety_car_laps,
        dnf=dnf,
        total_fuel_used=fuel_used,
        strategy_id=sid,
        race_result=result,
    )


def _validate_batch(strategy: Strategy, config: SimulationConfig, simulations: int) -> None:
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    validate_strategy(strategy, config)


def _baseline_strategy(config: SimulationConfig) -> Strategy:
    """Return the deterministic reference strategy used by non-focal drivers."""
    lap = max(1, min(config.track.number_of_laps - 1, config.track.number_of_laps // 2))
    return Strategy(TireCompound.MEDIUM, [lap], [TireCompound.HARD])


def simulate_strategy(
    config: SimulationConfig,
    strategy: Strategy,
    simulations: int,
    seed: int = 0,
) -> MonteCarloResult:
    """Evaluate a strategy on Driver 1 against a fixed baseline for the other drivers."""
    _validate_batch(strategy, config, simulations)
    sid = strategy_id(strategy)
    strategies = [strategy] + [_baseline_strategy(config)] * (len(config.drivers) - 1)
    seeds = tuple(int(seed + i) for i in range(simulations))
    races = tuple(_extract_race(simulate_race(config, strategies, s), config, sid) for s in seeds)
    return MonteCarloResult(strategy, sid, seeds, races, aggregate_statistics(races, sid, simulations))


def aggregate_statistics(
    races: Sequence[MonteCarloRace],
    sid: str,
    simulations: int | None = None,
) -> StrategyStatistics:
    """Compute aggregate statistics from Monte Carlo race outcomes."""
    if not races:
        raise ValueError("At least one race is required")
    n = simulations or len(races)
    if n != len(races):
        raise ValueError("simulations must match the number of race outcomes")
    times = np.asarray([r.total_race_time for r in races], dtype=float)
    wins = np.asarray([r.finishing_position[0] == 1 for r in races], dtype=float)
    podium = np.asarray([r.finishing_position[0] is not None and r.finishing_position[0] <= 3 for r in races], dtype=float)
    dnfs = np.asarray([r.dnf for r in races], dtype=float)
    std = float(np.std(times, ddof=0))
    # Lower time and lower variance are rewarded; this remains a simple benchmark score.
    risk_score = float(np.mean(times) + std)
    return StrategyStatistics(
        strategy_id=sid,
        simulations=n,
        mean_race_time=float(np.mean(times)),
        median_race_time=float(np.median(times)),
        std_race_time=std,
        percentile_5=float(np.percentile(times, 5)),
        percentile_95=float(np.percentile(times, 95)),
        win_probability=float(np.mean(wins)),
        podium_probability=float(np.mean(podium)),
        dnf_probability=float(np.mean(dnfs)),
        mean_pit_stops=float(np.mean([r.pit_stops for r in races])),
        mean_safety_car_laps=float(np.mean([r.safety_car_laps for r in races])),
        mean_fuel_used=float(np.mean([r.total_fuel_used for r in races])),
        risk_adjusted_score=risk_score,
    )


def race_driver_index(race: MonteCarloRace) -> int:
    """Return the index of the classified winner in a compact outcome."""
    for i, position in enumerate(race.finishing_position):
        if position == 1:
            return i
    return -1


def simulate_strategies_python(
    config: SimulationConfig,
    strategies: Sequence[Strategy],
    simulations_per_strategy: int,
    seed: int = 0,
) -> tuple[MonteCarloResult, ...]:
    """Run the straightforward Python Monte Carlo baseline."""
    return tuple(
        simulate_strategy(config, strategy, simulations_per_strategy, seed + i * simulations_per_strategy)
        for i, strategy in enumerate(strategies)
    )


def _vectorized_single_strategy(
    config: SimulationConfig,
    strategy: Strategy,
    simulations: int,
    seed: int,
) -> MonteCarloResult:
    """Run a batched NumPy approximation using the same Phase 1 equations."""
    _validate_batch(strategy, config, simulations)
    driver_strategies = [strategy] + [_baseline_strategy(config)] * (len(config.drivers) - 1)
    rng = np.random.default_rng(seed)
    d = len(config.drivers)
    laps = config.track.number_of_laps
    total = np.zeros((simulations, d), dtype=np.float64)
    fuel = np.ones((simulations, d), dtype=np.float64)
    tire_age = np.zeros((simulations, d), dtype=np.int32)
    dnf = np.zeros((simulations, d), dtype=bool)
    stops = np.zeros((simulations, d), dtype=np.int16)
    safety_laps = np.zeros(simulations, dtype=np.int16)
    fastest_lap = np.full((simulations, d), np.inf, dtype=np.float64)
    compounds = list(TireCompound)
    compound_index = {c.value: i for i, c in enumerate(compounds)}
    initial_ids = np.asarray([compound_index[s.initial_compound.value] for s in driver_strategies], dtype=np.int8)
    tire_ids = np.tile(initial_ids, (simulations, 1))
    seed_values = np.arange(seed, seed + simulations, dtype=np.int64)

    driver_skill = np.asarray([x.skill_rating for x in config.drivers])
    consistency = np.asarray([x.consistency for x in config.drivers])
    aggression = np.asarray([x.aggression for x in config.drivers])
    management = np.asarray([x.tire_management for x in config.drivers])
    car_perf = np.asarray([x.performance_coefficient for x in config.cars])
    car_reliability = np.asarray([x.reliability for x in config.cars])
    tire_perf = np.asarray([x.tire_performance_coefficient for x in config.cars])
    qualifying = np.asarray([x.qualifying_pace for x in config.drivers])
    initial_order = np.argsort(-qualifying)
    positions = np.tile(initial_order, (simulations, 1))
    weather = np.full(simulations, WeatherType.DRY.value, dtype=object)
    safety_remaining = np.zeros(simulations, dtype=np.int16)

    pace_delta = np.asarray([config.tires[c].pace_delta for c in compounds])
    degradation = np.asarray([config.tires[c].degradation_rate for c in compounds])
    warmup = np.asarray([config.tires[c].warmup_laps for c in compounds])
    useful = np.asarray([config.tires[c].max_useful_age for c in compounds])

    for lap in range(1, laps + 1):
        # Match the Phase 1 weather transition probabilities, batched across races.
        transition = rng.random(simulations) < config.weather.transition_probability
        draw = rng.random(simulations)
        dry = weather == WeatherType.DRY.value
        light = weather == WeatherType.LIGHT_RAIN.value
        heavy = weather == WeatherType.HEAVY_RAIN.value
        weather = np.where(transition & dry, np.where(draw < 0.75, WeatherType.LIGHT_RAIN.value, WeatherType.HEAVY_RAIN.value), weather)
        weather = np.where(transition & light, np.where(draw < 0.55, WeatherType.DRY.value, WeatherType.HEAVY_RAIN.value), weather)
        weather = np.where(transition & heavy, np.where(draw < 0.65, WeatherType.LIGHT_RAIN.value, WeatherType.DRY.value), weather)

        start_sc = (~(safety_remaining > 0)) & (rng.random(simulations) < config.weather.safety_car_probability)
        safety_remaining = np.where(start_sc, config.weather.safety_car_duration_laps, safety_remaining)
        sc_active = safety_remaining > 0

        # Scheduled stops reset age and switch compounds, exactly as Phase 1 does.
        for driver_index, driver_strategy in enumerate(driver_strategies):
            for stop_index, pit_lap in enumerate(driver_strategy.pit_laps):
                if lap == pit_lap:
                    new_id = compound_index[driver_strategy.compounds[stop_index].value]
                    active = ~dnf[:, driver_index]
                    total[active, driver_index] += config.track.pit_lane_time_loss
                    stops[active, driver_index] += 1
                    tire_ids[active, driver_index] = new_id
                    tire_age[active, driver_index] = 0

        tire_cfg_ids = tire_ids
        tire_pace = pace_delta[tire_cfg_ids]
        tire_deg = degradation[tire_cfg_ids]
        tire_warm = warmup[tire_cfg_ids]
        age = tire_age.astype(float)
        management_factor = 1.0 - 0.35 * np.clip(management[None, :], 0, 1)
        degradation_penalty = tire_deg * np.power(age, 1.35) * management_factor * config.track.tire_degradation_multiplier
        warmup_penalty = np.where((tire_warm > 0) & (age < tire_warm), (tire_warm - age) / tire_warm * 0.8, 0.0)
        fuel_penalty = fuel * config.fuel_effect_per_lap
        weather_base = np.select(
            [weather == WeatherType.LIGHT_RAIN.value, weather == WeatherType.HEAVY_RAIN.value],
            [config.weather.light_rain_penalty, config.weather.heavy_rain_penalty],
            default=config.weather.dry_penalty,
        )[:, None]
        mismatch = np.zeros((simulations, d), dtype=float)
        mismatch_table = {
            WeatherType.DRY.value: [0.0, 0.0, 0.0, 9.0, 15.0],
            WeatherType.LIGHT_RAIN.value: [4.0, 3.0, 3.5, 0.0, 2.0],
            WeatherType.HEAVY_RAIN.value: [11.0, 10.0, 10.0, 2.0, 0.0],
        }
        for weather_name, values in mismatch_table.items():
            mask = weather == weather_name
            mismatch[mask] = np.asarray(values)[tire_cfg_ids[mask]]

        lap_time = (
            config.track.base_lap_time
            + (0.5 - driver_skill)[None, :] * 2.4
            + (1.0 - car_perf)[None, :] * 1.8
            - tire_pace * tire_perf[None, :]
            + degradation_penalty
            + warmup_penalty
            + fuel_penalty
            + weather_base
            + mismatch
        )

        # Traffic is evaluated from the previous lap's order, matching Phase 1's dependency structure.
        order = positions
        ranked_total = np.take_along_axis(total, order, axis=1)
        gap_ahead = np.zeros((simulations, d), dtype=float)
        if d > 1:
            ahead_time = np.zeros_like(ranked_total)
            ahead_time[:, 1:] = ranked_total[:, :-1]
            ranked_gap = np.maximum(0.0, ranked_total - ahead_time)
            ranked_penalty = config.traffic_base_penalty * np.maximum(0.0, 1.0 - ranked_gap / 3.0) * config.track.overtaking_difficulty
            ranked_penalty *= (1.0 - 0.35 * aggression[None, :])
            ranked_penalty[:, 0] = 0.0
            inverse = np.argsort(order, axis=1)
            traffic = np.take_along_axis(ranked_penalty, inverse, axis=1)
            lap_time += traffic

        noise_scale = config.stochastic_noise_std * (1.2 - consistency)[None, :]
        lap_time += rng.normal(0.0, 1.0, size=(simulations, d)) * noise_scale
        lap_time = np.maximum(1.0, lap_time)
        if np.any(sc_active):
            lap_time[sc_active] = 0.88 * lap_time[sc_active] + 0.12 * config.track.base_lap_time
            safety_laps[sc_active] += 1

        failure_probability = config.dnf_base_probability + (1.0 - car_reliability)[None, :] * 0.012
        failures = (rng.random((simulations, d)) < failure_probability) & ~dnf
        dnf |= failures
        lap_time[dnf] = 0.0
        total += lap_time
        fastest_lap = np.minimum(fastest_lap, np.where(dnf, np.inf, lap_time))
        tire_age[~dnf] += 1
        fuel = np.maximum(0.0, fuel - 1.0 / laps)
        safety_remaining = np.maximum(0, safety_remaining - 1)
        positions = np.argsort(np.where(dnf, np.inf, total), axis=1)

    race_times = np.max(np.where(dnf, np.nan, total), axis=1)
    race_times = np.where(np.isnan(race_times), 0.0, race_times)
    results: list[MonteCarloRace] = []
    sid = strategy_id(strategy)
    for i in range(simulations):
        ordered = positions[i]
        pos = [None] * d
        for p, driver_idx in enumerate(ordered, start=1):
            if not dnf[i, driver_idx]:
                pos[int(driver_idx)] = p
        fastest = float(np.min(fastest_lap[i])) if np.isfinite(np.min(fastest_lap[i])) else None
        results.append(MonteCarloRace(
            seed=int(seed_values[i]),
            winner=config.drivers[int(ordered[0])].name if not dnf[i, ordered[0]] else None,
            finishing_position=tuple(pos),
            total_race_time=float(race_times[i]),
            fastest_lap=fastest,
            pit_stops=int(stops[i].sum()),
            tire_usage=tuple(c.value for c in compounds if np.any(tire_ids[i] == compound_index[c.value])),
            safety_car_laps=int(safety_laps[i]),
            dnf=bool(np.any(dnf[i])),
            total_fuel_used=float(np.sum(1.0 - fuel[i])),
            strategy_id=sid,
            race_result=None,
        ))
    return MonteCarloResult(strategy, sid, tuple(int(x) for x in seed_values), tuple(results), aggregate_statistics(results, sid))


def simulate_strategies_numpy(
    config: SimulationConfig,
    strategies: Sequence[Strategy],
    simulations_per_strategy: int,
    seed: int = 0,
) -> tuple[MonteCarloResult, ...]:
    """Run the batched NumPy Monte Carlo baseline."""
    if simulations_per_strategy <= 0:
        raise ValueError("simulations_per_strategy must be positive")
    return tuple(
        _vectorized_single_strategy(config, strategy, simulations_per_strategy, seed + i * simulations_per_strategy)
        for i, strategy in enumerate(strategies)
    )


def benchmark_engine(
    engine: Callable[..., Sequence[MonteCarloResult]],
    config: SimulationConfig,
    strategies: Sequence[Strategy],
    simulations_per_strategy: int,
    seed: int = 0,
) -> BenchmarkResult:
    """Time one Monte Carlo engine under standardized inputs."""
    start = perf_counter()
    engine(config, strategies, simulations_per_strategy, seed)
    elapsed = perf_counter() - start
    total = len(strategies) * simulations_per_strategy
    return BenchmarkResult(
        len(strategies), simulations_per_strategy, total, elapsed, total / elapsed if elapsed else float("inf")
    )
