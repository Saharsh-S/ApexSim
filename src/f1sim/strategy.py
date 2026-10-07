"""Strategy validation and helpers."""

from __future__ import annotations

from .config import SimulationConfig
from .models import Strategy


def validate_strategy(strategy: Strategy, config: SimulationConfig) -> None:
    """Validate a strategy against a race configuration."""
    if any(lap >= config.track.number_of_laps for lap in strategy.pit_laps):
        raise ValueError("Pit laps must occur before the final race lap")
    for compound in strategy.compounds:
        if compound not in config.tires:
            raise ValueError(f"Unsupported tire compound: {compound}")


def strategy_for_lap(strategy: Strategy, lap: int) -> int | None:
    """Return the stop index scheduled for a given completed lap, if any."""
    try:
        return strategy.pit_laps.index(lap)
    except ValueError:
        return None
