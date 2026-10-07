"""Run a small example race."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1sim import (
    CarConfig, DriverConfig, SimulationConfig, Strategy, TireCompound, TireConfig,
    TrackConfig, WeatherConfig, simulate_race,
)


def build_config() -> SimulationConfig:
    tires = {
        TireCompound.SOFT: TireConfig(TireCompound.SOFT, 1.2, 0.070, 1.0, 18),
        TireCompound.MEDIUM: TireConfig(TireCompound.MEDIUM, 0.6, 0.045, 1.0, 28),
        TireCompound.HARD: TireConfig(TireCompound.HARD, 0.1, 0.025, 2.0, 38),
        TireCompound.INTERMEDIATE: TireConfig(TireCompound.INTERMEDIATE, -0.4, 0.035, 1.0, 25),
        TireCompound.WET: TireConfig(TireCompound.WET, -0.2, 0.025, 1.0, 30),
    }
    drivers = [
        DriverConfig("Driver 1", 0.94, 0.90, 0.75, 0.80, 0.96),
        DriverConfig("Driver 2", 0.90, 0.86, 0.65, 0.75, 0.92),
        DriverConfig("Driver 3", 0.86, 0.82, 0.80, 0.70, 0.88),
        DriverConfig("Driver 4", 0.82, 0.88, 0.55, 0.85, 0.84),
    ]
    cars = [
        CarConfig("Team A", 0.97, 0.995, 1.02),
        CarConfig("Team B", 0.94, 0.992, 1.00),
        CarConfig("Team C", 0.91, 0.985, 0.99),
        CarConfig("Team D", 0.88, 0.98, 0.98),
    ]
    return SimulationConfig(
        TrackConfig("Reference Ring", 30, 5.2, 92.0, 22.0, 0.55),
        drivers,
        cars,
        tires,
        WeatherConfig(safety_car_probability=0.02),
    )


if __name__ == "__main__":
    config = build_config()
    strategy = Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD])
    result = simulate_race(config, strategy, seed=42)
    for driver in result.driver_results:
        status = "DNF" if driver.dnf else f"P{driver.finishing_position}"
        print(f"{status:>3} {driver.driver:10s}  time={driver.total_time!s:>8}  stops={driver.pit_stops}")
    print(f"Safety-car laps: {result.safety_car_laps}")
