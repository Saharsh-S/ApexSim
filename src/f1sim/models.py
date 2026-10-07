"""Runtime models and result structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import TireCompound, WeatherType


@dataclass
class Strategy:
    """A fixed race strategy consisting of a starting tire and planned pit laps."""

    initial_compound: TireCompound
    pit_laps: list[int] = field(default_factory=list)
    compounds: list[TireCompound] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.pit_laps) != len(self.compounds):
            raise ValueError("pit_laps and compounds must have equal length")
        if any(lap <= 0 for lap in self.pit_laps):
            raise ValueError("Pit laps must be positive")
        if self.pit_laps != sorted(self.pit_laps) or len(set(self.pit_laps)) != len(self.pit_laps):
            raise ValueError("pit_laps must be strictly increasing")

    @property
    def number_of_stops(self) -> int:
        return len(self.pit_laps)

    def compound_for_stop(self, stop_index: int) -> TireCompound:
        return self.compounds[stop_index]


@dataclass
class DriverRaceState:
    """Mutable per-driver race state."""

    index: int
    name: str
    current_tire: TireCompound
    tire_age: int = 0
    fuel: float = 1.0
    total_time: float = 0.0
    total_pit_time: float = 0.0
    pit_stops: int = 0
    dnf: bool = False
    lap_times: list[float] = field(default_factory=list)
    tire_history: list[TireCompound] = field(default_factory=list)


@dataclass
class RaceState:
    """Complete mutable state at a point in the race."""

    current_lap: int
    current_time: float
    drivers: list[DriverRaceState]
    positions: list[int]
    weather: WeatherType
    safety_car_active: bool = False
    safety_car_laps_remaining: int = 0
    safety_car_laps: int = 0


@dataclass(frozen=True)
class DriverResult:
    """Final result for one driver."""

    driver: str
    finishing_position: Optional[int]
    total_time: Optional[float]
    total_pit_time: float
    pit_stops: int
    tire_usage: tuple[TireCompound, ...]
    lap_times: tuple[float, ...]
    fastest_lap: Optional[float]
    average_lap: Optional[float]
    dnf: bool


@dataclass(frozen=True)
class RaceResult:
    """Structured output of a completed race."""

    track: str
    seed: int
    total_race_time: float
    safety_car_laps: int
    driver_results: tuple[DriverResult, ...]
    positions_by_lap: tuple[tuple[int, ...], ...]
    weather_by_lap: tuple[WeatherType, ...]

    def winner(self) -> Optional[DriverResult]:
        """Return the classified winner, if one exists."""
        for result in self.driver_results:
            if result.finishing_position == 1:
                return result
        return None
