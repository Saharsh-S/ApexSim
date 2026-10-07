import numpy as np
from f1sim.config import WeatherConfig, WeatherType
from f1sim.weather import transition_weather, weather_penalty


def test_zero_transition_probability_is_deterministic():
    config = WeatherConfig(transition_probability=0)
    rng = np.random.default_rng(42)
    assert transition_weather(WeatherType.DRY, config, rng) == WeatherType.DRY


def test_weather_penalty_order():
    config = WeatherConfig()
    assert weather_penalty(WeatherType.DRY, config) < weather_penalty(WeatherType.LIGHT_RAIN, config)
    assert weather_penalty(WeatherType.LIGHT_RAIN, config) < weather_penalty(WeatherType.HEAVY_RAIN, config)
