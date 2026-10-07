from dataclasses import replace
import pytest

from f1sim import Strategy, TireCompound, simulate_race
from examples.basic_race import build_config


def test_same_seed_produces_identical_result():
    config = build_config()
    strategy = Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD])
    first = simulate_race(config, strategy, seed=123)
    second = simulate_race(config, strategy, seed=123)
    assert first == second


def test_different_seed_can_change_result():
    config = build_config()
    strategy = Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD])
    first = simulate_race(config, strategy, seed=123)
    second = simulate_race(config, strategy, seed=124)
    assert first != second


def test_invalid_configuration_rejected():
    config = build_config()
    with pytest.raises(ValueError):
        replace(config.track, number_of_laps=0)
