from f1sim.config import TireCompound, TireConfig
from f1sim.tires import tire_degradation_penalty, tire_warmup_penalty, tire_is_past_useful_age


def test_degradation_increases_with_age():
    tire = TireConfig(TireCompound.SOFT, 1.2, 0.07, 1, 18)
    assert tire_degradation_penalty(tire, 10) > tire_degradation_penalty(tire, 2)


def test_tire_management_reduces_degradation():
    tire = TireConfig(TireCompound.MEDIUM, 0.5, 0.05, 1, 25)
    assert tire_degradation_penalty(tire, 12, 1.0) < tire_degradation_penalty(tire, 12, 0.0)


def test_warmup_penalty_disappears_after_warmup():
    tire = TireConfig(TireCompound.HARD, 0.1, 0.02, 2, 40)
    assert tire_warmup_penalty(tire, 0) > 0
    assert tire_warmup_penalty(tire, 2) == 0


def test_useful_age():
    tire = TireConfig(TireCompound.SOFT, 1, 0.1, 1, 10)
    assert not tire_is_past_useful_age(tire, 10)
    assert tire_is_past_useful_age(tire, 11)
