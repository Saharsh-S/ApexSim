from examples.basic_race import build_config
from f1sim.config import TireCompound, WeatherType
from f1sim.realtime import RaceState, RealtimeStrategyEngine


def test_realtime_decision_is_deterministic():
    cfg = build_config()
    state = RaceState(37, 4, TireCompound.MEDIUM, 18, 2.4, 1.8, .62, .15, .42, WeatherType.DRY)
    engine = RealtimeStrategyEngine(cfg, seed=42)
    a = engine.decide(state, candidate_count=16, simulations_per_candidate=4)
    b = engine.decide(state, candidate_count=16, simulations_per_candidate=4)
    assert a.recommendation.strategy == b.recommendation.strategy
    assert a.recommendation.expected_finish == b.recommendation.expected_finish


def test_realtime_has_stage_timings():
    cfg = build_config()
    state = RaceState(37, 4, TireCompound.MEDIUM, 18, 2.4, 1.8, .62, .15)
    d = RealtimeStrategyEngine(cfg).decide(state, candidate_count=8, simulations_per_candidate=2)
    assert d.candidates_evaluated == 8
    assert d.timing.total_ms >= d.timing.evaluation_ms
    assert d.recommendation.rank == 1
    assert len(d.alternatives) == 2
