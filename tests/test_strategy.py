import pytest
from f1sim import Strategy, TireCompound
from examples.basic_race import build_config
from f1sim.strategy import validate_strategy


def test_strategy_validation():
    config = build_config()
    validate_strategy(Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD]), config)


def test_strategy_rejects_late_stop():
    config = build_config()
    with pytest.raises(ValueError):
        validate_strategy(Strategy(TireCompound.MEDIUM, [30], [TireCompound.HARD]), config)


def test_strategy_rejects_mismatched_lists():
    with pytest.raises(ValueError):
        Strategy(TireCompound.MEDIUM, [10], [])
