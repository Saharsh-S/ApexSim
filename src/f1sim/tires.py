"""Tire performance calculations."""

from __future__ import annotations

from .config import TireConfig, TireCompound


def tire_degradation_penalty(
    tire: TireConfig,
    age: int,
    tire_management: float = 0.5,
    track_multiplier: float = 1.0,
) -> float:
    """Return seconds lost from degradation for a tire of a given age."""
    if age < 0:
        raise ValueError("Tire age cannot be negative")
    management_factor = 1.0 - 0.35 * max(0.0, min(1.0, tire_management))
    return tire.degradation_rate * (age**1.35) * management_factor * track_multiplier


def tire_warmup_penalty(tire: TireConfig, age: int) -> float:
    """Return the temporary warmup penalty for a new tire."""
    if age < 0:
        raise ValueError("Tire age cannot be negative")
    if tire.warmup_laps == 0 or age >= tire.warmup_laps:
        return 0.0
    return (tire.warmup_laps - age) / tire.warmup_laps * 0.8


def tire_is_past_useful_age(tire: TireConfig, age: int) -> bool:
    """Whether the tire has exceeded its configured useful life."""
    return age > tire.max_useful_age


def compound_name(compound: TireCompound) -> str:
    """Return a display-friendly compound name."""
    return compound.value.title()
