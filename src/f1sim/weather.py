"""Weather and safety-car models."""

from __future__ import annotations

import numpy as np

from .config import WeatherConfig, WeatherType


def transition_weather(current: WeatherType, config: WeatherConfig, rng: np.random.Generator) -> WeatherType:
    """Sample a simple one-step weather transition."""
    if rng.random() >= config.transition_probability:
        return current
    if current == WeatherType.DRY:
        return WeatherType.LIGHT_RAIN if rng.random() < 0.75 else WeatherType.HEAVY_RAIN
    if current == WeatherType.LIGHT_RAIN:
        return WeatherType.DRY if rng.random() < 0.55 else WeatherType.HEAVY_RAIN
    return WeatherType.LIGHT_RAIN if rng.random() < 0.65 else WeatherType.DRY


def weather_penalty(weather: WeatherType, config: WeatherConfig) -> float:
    """Return baseline lap-time penalty for the current weather."""
    return {
        WeatherType.DRY: config.dry_penalty,
        WeatherType.LIGHT_RAIN: config.light_rain_penalty,
        WeatherType.HEAVY_RAIN: config.heavy_rain_penalty,
    }[weather]


def should_start_safety_car(config: WeatherConfig, rng: np.random.Generator) -> bool:
    """Sample whether a safety-car event begins on this lap."""
    return bool(rng.random() < config.safety_car_probability)
