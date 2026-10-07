"""Configuration objects for the F1 race simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class TireCompound(str, Enum):
    """Supported tire compounds."""

    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"


class WeatherType(str, Enum):
    """Supported weather states."""

    DRY = "dry"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"


@dataclass(frozen=True)
class TireConfig:
    """Tire performance and degradation parameters."""

    compound: TireCompound
    pace_delta: float
    degradation_rate: float
    warmup_laps: float
    max_useful_age: int

    def __post_init__(self) -> None:
        if self.degradation_rate < 0 or self.warmup_laps < 0:
            raise ValueError("Tire degradation and warmup must be non-negative")
        if self.max_useful_age <= 0:
            raise ValueError("max_useful_age must be positive")


@dataclass(frozen=True)
class TrackConfig:
    """Track-level simulation parameters."""

    name: str
    number_of_laps: int
    lap_distance_km: float
    base_lap_time: float
    pit_lane_time_loss: float
    overtaking_difficulty: float
    tire_degradation_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.number_of_laps <= 0 or self.lap_distance_km <= 0:
            raise ValueError("Track distance and lap count must be positive")
        if self.base_lap_time <= 0 or self.pit_lane_time_loss < 0:
            raise ValueError("Track timing parameters are invalid")
        if not 0 <= self.overtaking_difficulty <= 1:
            raise ValueError("overtaking_difficulty must be between 0 and 1")
        if self.tire_degradation_multiplier <= 0:
            raise ValueError("tire_degradation_multiplier must be positive")


@dataclass(frozen=True)
class DriverConfig:
    """Driver characteristics."""

    name: str
    skill_rating: float
    consistency: float
    aggression: float
    tire_management: float
    qualifying_pace: float

    def __post_init__(self) -> None:
        for name, value in (("skill_rating", self.skill_rating), ("consistency", self.consistency),
                            ("aggression", self.aggression), ("tire_management", self.tire_management),
                            ("qualifying_pace", self.qualifying_pace)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class CarConfig:
    """Car/team characteristics."""

    team: str
    performance_coefficient: float
    reliability: float
    tire_performance_coefficient: float = 1.0

    def __post_init__(self) -> None:
        if self.performance_coefficient <= 0 or self.tire_performance_coefficient <= 0:
            raise ValueError("Car performance coefficients must be positive")
        if not 0 <= self.reliability <= 1:
            raise ValueError("reliability must be between 0 and 1")


@dataclass(frozen=True)
class WeatherConfig:
    """Weather transition and lap-time parameters."""

    initial: WeatherType = WeatherType.DRY
    transition_probability: float = 0.03
    dry_penalty: float = 0.0
    light_rain_penalty: float = 7.0
    heavy_rain_penalty: float = 16.0
    safety_car_probability: float = 0.015
    safety_car_duration_laps: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.transition_probability <= 1:
            raise ValueError("transition_probability must be between 0 and 1")
        if not 0 <= self.safety_car_probability <= 1:
            raise ValueError("safety_car_probability must be between 0 and 1")
        if self.safety_car_duration_laps <= 0:
            raise ValueError("safety_car_duration_laps must be positive")


@dataclass(frozen=True)
class SimulationConfig:
    """Complete simulator configuration."""

    track: TrackConfig
    drivers: Sequence[DriverConfig]
    cars: Sequence[CarConfig]
    tires: Mapping[TireCompound, TireConfig]
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    fuel_effect_per_lap: float = 0.035
    traffic_base_penalty: float = 0.6
    stochastic_noise_std: float = 0.10
    dnf_base_probability: float = 0.001
    safety_car_gap_compression: float = 0.65

    def __post_init__(self) -> None:
        if len(self.drivers) == 0:
            raise ValueError("At least one driver is required")
        if len(self.drivers) != len(self.cars):
            raise ValueError("drivers and cars must have the same length")
        required = set(TireCompound)
        if set(self.tires) != required:
            raise ValueError("tires must define all five tire compounds")
        if self.fuel_effect_per_lap < 0 or self.traffic_base_penalty < 0 or self.stochastic_noise_std < 0:
            raise ValueError("Simulation penalties and noise must be non-negative")
        if not 0 <= self.dnf_base_probability <= 1:
            raise ValueError("dnf_base_probability must be between 0 and 1")
        if not 0 <= self.safety_car_gap_compression <= 1:
            raise ValueError("safety_car_gap_compression must be between 0 and 1")
