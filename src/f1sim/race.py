"""Core CPU race engine."""

from __future__ import annotations

import numpy as np

from .config import SimulationConfig, TireCompound, WeatherType
from .metrics import compute_traffic_penalty, rank_drivers
from .models import DriverRaceState, RaceResult, RaceState, Strategy, DriverResult
from .strategy import strategy_for_lap, validate_strategy
from .tires import tire_degradation_penalty, tire_warmup_penalty
from .weather import should_start_safety_car, transition_weather, weather_penalty


def _driver_delta(skill: float) -> float:
    return (0.5 - skill) * 2.4


def _car_delta(performance: float) -> float:
    return (1.0 - performance) * 1.8


def _tire_delta(config: SimulationConfig, compound: TireCompound, car_coefficient: float) -> float:
    return -config.tires[compound].pace_delta * car_coefficient


def _fuel_penalty(fuel: float, fuel_effect_per_lap: float) -> float:
    return fuel * fuel_effect_per_lap


def _weather_tire_mismatch(weather: WeatherType, compound: TireCompound) -> float:
    if weather == WeatherType.DRY:
        return {TireCompound.SOFT: 0.0, TireCompound.MEDIUM: 0.0, TireCompound.HARD: 0.0,
                TireCompound.INTERMEDIATE: 9.0, TireCompound.WET: 15.0}[compound]
    if weather == WeatherType.LIGHT_RAIN:
        return {TireCompound.SOFT: 4.0, TireCompound.MEDIUM: 3.0, TireCompound.HARD: 3.5,
                TireCompound.INTERMEDIATE: 0.0, TireCompound.WET: 2.0}[compound]
    return {TireCompound.SOFT: 11.0, TireCompound.MEDIUM: 10.0, TireCompound.HARD: 10.0,
            TireCompound.INTERMEDIATE: 2.0, TireCompound.WET: 0.0}[compound]


def _initialize_state(config: SimulationConfig, strategies: list[Strategy]) -> RaceState:
    drivers = []
    for i, (driver, strategy) in enumerate(zip(config.drivers, strategies)):
        drivers.append(
            DriverRaceState(
                index=i,
                name=driver.name,
                current_tire=strategy.initial_compound,
                tire_history=[strategy.initial_compound],
            )
        )
    initial_order = sorted(
        range(len(drivers)),
        key=lambda i: (-config.drivers[i].qualifying_pace, i),
    )
    return RaceState(
        current_lap=0,
        current_time=0.0,
        drivers=drivers,
        positions=initial_order,
        weather=config.weather.initial,
    )


def _apply_pit_stop(driver: DriverRaceState, strategy: Strategy, config: SimulationConfig, lap: int) -> None:
    stop_index = strategy_for_lap(strategy, lap)
    if stop_index is None:
        return
    driver.current_tire = strategy.compound_for_stop(stop_index)
    driver.tire_age = 0
    driver.total_pit_time += config.track.pit_lane_time_loss
    driver.total_time += config.track.pit_lane_time_loss
    driver.pit_stops += 1
    driver.tire_history.append(driver.current_tire)


def _simulate_lap_time(
    driver: DriverRaceState,
    driver_config,
    car_config,
    config: SimulationConfig,
    state: RaceState,
    rng: np.random.Generator,
    position: int,
) -> float:
    tire = config.tires[driver.current_tire]
    lap_time = config.track.base_lap_time
    lap_time += _driver_delta(driver_config.skill_rating)
    lap_time += _car_delta(car_config.performance_coefficient)
    lap_time += _tire_delta(config, driver.current_tire, car_config.tire_performance_coefficient)
    lap_time += tire_degradation_penalty(
        tire,
        driver.tire_age,
        driver_config.tire_management,
        config.track.tire_degradation_multiplier,
    )
    lap_time += tire_warmup_penalty(tire, driver.tire_age)
    lap_time += _fuel_penalty(driver.fuel, config.fuel_effect_per_lap)
    lap_time += weather_penalty(state.weather, config.weather)
    lap_time += _weather_tire_mismatch(state.weather, driver.current_tire)

    gap_ahead = 10.0
    if position > 1:
        ahead = state.positions[position - 2]
        gap_ahead = max(0.0, driver.total_time - state.drivers[ahead].total_time)
    lap_time += compute_traffic_penalty(
        position,
        gap_ahead,
        config.track.overtaking_difficulty,
        driver_config.aggression,
        config.traffic_base_penalty,
    )

    if state.safety_car_active:
        lap_time = 0.88 * lap_time + 0.12 * config.track.base_lap_time
    noise_scale = config.stochastic_noise_std * (1.2 - driver_config.consistency)
    lap_time += rng.normal(0.0, noise_scale)
    return max(1.0, lap_time)


def _maybe_dnf(driver: DriverRaceState, car_config, config: SimulationConfig, rng: np.random.Generator) -> bool:
    # Less reliable cars and random race incidents increase the failure probability.
    probability = config.dnf_base_probability + (1.0 - car_config.reliability) * 0.012
    return bool(rng.random() < probability)


def simulate_race(config: SimulationConfig, strategies: list[Strategy], seed: int = 0) -> RaceResult:
    """Simulate one race deterministically from a seed and return structured results."""
    if len(strategies) != len(config.drivers):
        raise ValueError("One strategy is required for each driver")
    for strategy in strategies:
        validate_strategy(strategy, config)

    rng = np.random.default_rng(seed)
    state = _initialize_state(config, strategies)
    positions_by_lap: list[tuple[int, ...]] = []
    weather_by_lap: list[WeatherType] = []

    for lap in range(1, config.track.number_of_laps + 1):
        state.current_lap = lap
        state.weather = transition_weather(state.weather, config.weather, rng)
        if not state.safety_car_active and should_start_safety_car(config.weather, rng):
            state.safety_car_active = True
            state.safety_car_laps_remaining = config.weather.safety_car_duration_laps

        # Pit before lap-time calculation so the stop affects this lap's total race time.
        for driver, strategy in zip(state.drivers, strategies):
            if not driver.dnf:
                _apply_pit_stop(driver, strategy, config, lap)

        state.positions = rank_drivers(state.drivers)
        for position, driver_index in enumerate(state.positions, start=1):
            driver = state.drivers[driver_index]
            if driver.dnf:
                continue
            if _maybe_dnf(driver, config.cars[driver.index], config, rng):
                driver.dnf = True
                continue
            lap_time = _simulate_lap_time(
                driver,
                config.drivers[driver.index],
                config.cars[driver.index],
                config,
                state,
                rng,
                position,
            )
            driver.total_time += lap_time
            driver.lap_times.append(lap_time)
            driver.tire_age += 1
            driver.fuel = max(0.0, driver.fuel - 1.0 / config.track.number_of_laps)

        state.positions = rank_drivers(state.drivers)
        if state.safety_car_active:
            state.safety_car_laps += 1
            state.safety_car_laps_remaining -= 1
            if state.safety_car_laps_remaining <= 0:
                state.safety_car_active = False
        positions_by_lap.append(tuple(state.positions))
        weather_by_lap.append(state.weather)

    classified = [d for d in state.drivers if not d.dnf]
    classified.sort(key=lambda d: d.total_time)
    finishing_positions = {d.index: i + 1 for i, d in enumerate(classified)}
    total_race_time = max((d.total_time for d in classified), default=0.0)

    results = []
    for driver in state.drivers:
        results.append(
            DriverResult(
                driver=driver.name,
                finishing_position=finishing_positions.get(driver.index),
                total_time=driver.total_time if not driver.dnf else None,
                total_pit_time=driver.total_pit_time,
                pit_stops=driver.pit_stops,
                tire_usage=tuple(driver.tire_history),
                lap_times=tuple(driver.lap_times),
                fastest_lap=min(driver.lap_times) if driver.lap_times else None,
                average_lap=float(np.mean(driver.lap_times)) if driver.lap_times else None,
                dnf=driver.dnf,
            )
        )
    results.sort(key=lambda r: (r.finishing_position is None, r.finishing_position or 999, r.driver))

    return RaceResult(
        track=config.track.name,
        seed=seed,
        total_race_time=total_race_time,
        safety_car_laps=state.safety_car_laps,
        driver_results=tuple(results),
        positions_by_lap=tuple(positions_by_lap),
        weather_by_lap=tuple(weather_by_lap),
    )
