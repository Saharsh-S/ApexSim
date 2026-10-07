import numpy as np
import pytest

from examples.basic_race import build_config
from f1sim import Strategy, TireCompound
from f1sim.analysis import convergence_curve, rank_strategies
from f1sim.monte_carlo import (
    aggregate_statistics,
    simulate_strategy,
    simulate_strategies_numpy,
    simulate_strategies_python,
)
from f1sim.strategy_generator import StrategyGenerationConfig, generate_strategies


@pytest.fixture
def config():
    return build_config()


def medium_hard():
    return Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD])


def test_monte_carlo_contains_valid_races(config):
    result = simulate_strategy(config, medium_hard(), 20, seed=10)
    assert len(result.races) == 20
    assert all(r.total_race_time > 0 for r in result.races)
    assert all(r.strategy_id == result.strategy_id for r in result.races)
    assert 0 <= result.statistics.win_probability <= 1


def test_monte_carlo_is_deterministic(config):
    a = simulate_strategy(config, medium_hard(), 25, seed=123)
    b = simulate_strategy(config, medium_hard(), 25, seed=123)
    assert a == b


def test_strategy_results_are_reproducible_across_engines(config):
    strategies = [medium_hard()]
    python_result = simulate_strategies_python(config, strategies, 5, seed=7)[0]
    numpy_result = simulate_strategies_numpy(config, strategies, 5, seed=7)[0]
    assert python_result.seeds == numpy_result.seeds
    assert all(r.strategy_id == numpy_result.strategy_id for r in numpy_result.races)
    assert len(python_result.races) == len(numpy_result.races) == 5


def test_invalid_strategy_fails(config):
    with pytest.raises(ValueError):
        simulate_strategy(config, Strategy(TireCompound.MEDIUM, [config.track.number_of_laps], [TireCompound.HARD]), 2)


def test_aggregate_statistics(config):
    result = simulate_strategy(config, medium_hard(), 50, seed=1)
    stats = aggregate_statistics(result.races, result.strategy_id)
    times = np.asarray([r.total_race_time for r in result.races])
    assert stats.mean_race_time == pytest.approx(float(np.mean(times)))
    assert stats.median_race_time == pytest.approx(float(np.median(times)))
    assert stats.percentile_5 <= stats.median_race_time <= stats.percentile_95


def test_strategy_generator_produces_valid_unique_strategies(config):
    strategies = generate_strategies(config, StrategyGenerationConfig(max_strategies=20))
    assert 0 < len(strategies) <= 20
    keys = {(s.initial_compound, tuple(s.pit_laps), tuple(s.compounds)) for s in strategies}
    assert len(keys) == len(strategies)
    for strategy in strategies:
        assert len(strategy.pit_laps) == len(strategy.compounds)
        assert all(lap < config.track.number_of_laps for lap in strategy.pit_laps)


def test_ranking_and_convergence(config):
    results = simulate_strategies_numpy(config, [medium_hard(), Strategy(TireCompound.SOFT, [10], [TireCompound.MEDIUM])], 30, seed=2)
    ranked = rank_strategies(results, by="win_probability")
    assert len(ranked) == 2
    counts, probabilities = convergence_curve(results[0])
    assert len(counts) == len(probabilities) == 30
    assert probabilities[-1] == pytest.approx(results[0].statistics.win_probability)


def test_aggregate_statistics_tracks_focal_driver_not_sorted_winner():
    """Win/podium rates must use Driver 1's result, not the sorted winner at index 0."""
    from f1sim.monte_carlo import MonteCarloRace

    races = tuple(
        MonteCarloRace(
            seed=i,
            winner="Driver 1" if pos == 1 else "Driver 2",
            finishing_position=positions,
            total_race_time=100.0 + i,
            fastest_lap=None,
            pit_stops=1,
            tire_usage=("medium",),
            safety_car_laps=0,
            dnf=False,
            total_fuel_used=4.0,
            strategy_id="medium@15-hard",
            race_result=None,
        )
        for i, (pos, positions) in enumerate([
            (1, (1, 2, 3, 4)),
            (2, (2, 1, 3, 4)),
            (4, (4, 1, 2, 3)),
        ])
    )
    stats = aggregate_statistics(races, "medium@15-hard")
    assert stats.win_probability == pytest.approx(1 / 3)
    assert stats.podium_probability == pytest.approx(2 / 3)
