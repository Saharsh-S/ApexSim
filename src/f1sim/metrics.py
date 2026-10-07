"""Race metrics and position calculations."""

from __future__ import annotations

from .models import DriverRaceState


def rank_drivers(drivers: list[DriverRaceState]) -> list[int]:
    """Return driver indices ordered by race progress, excluding DNFs last."""
    return [
        driver.index
        for driver in sorted(
            drivers,
            key=lambda d: (d.dnf, d.total_time),
        )
    ]


def compute_traffic_penalty(
    driver_position: int,
    gap_ahead: float,
    overtaking_difficulty: float,
    aggression: float,
    base_penalty: float,
) -> float:
    """Estimate time lost in traffic based on gap and overtaking difficulty."""
    if driver_position == 1 or gap_ahead > 3.0:
        return 0.0
    pressure = max(0.0, 1.0 - gap_ahead / 3.0)
    aggression_relief = 1.0 - 0.35 * aggression
    return base_penalty * pressure * overtaking_difficulty * aggression_relief
