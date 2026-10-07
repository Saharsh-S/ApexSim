from __future__ import annotations

import numpy as np
import pytest

from examples.basic_race import build_config
from f1sim import StrategySearchConfig, generate_search_candidates, search_cpu, summarize_top_strategies


def test_candidate_generation_is_deterministic_and_valid():
    config=build_config()
    cfg=StrategySearchConfig(candidate_count=40, simulations_per_strategy=3, seed=123)
    a=generate_search_candidates(config,cfg); b=generate_search_candidates(config,cfg)
    assert [x.strategy_id for x in a] == [x.strategy_id for x in b]
    assert len(a)==40
    assert all(x.strategy.number_of_stops in (1,2) for x in a)


def test_search_numpy_returns_top_ten_metrics():
    config=build_config(); cfg=StrategySearchConfig(candidate_count=20, simulations_per_strategy=5, seed=7)
    candidates=generate_search_candidates(config,cfg)
    result=search_cpu(config,candidates,5,7,engine='numpy',top_k=10)
    top=summarize_top_strategies(result)
    assert len(top)==10
    assert all(1 <= x.expected_finishing_position <= len(config.drivers)+1 for x in top)
    assert all(0 <= x.win_probability <= 1 for x in top)
    assert all(x.expected_pit_stops >= 0 for x in top)


def test_search_count_and_throughput_metrics():
    config=build_config(); cfg=StrategySearchConfig(candidate_count=5, simulations_per_strategy=4, seed=9)
    result=search_cpu(config,generate_search_candidates(config,cfg),4,9,engine='numpy')
    assert result.total_simulations == 20
    assert result.strategies_per_second > 0
    assert result.simulations_per_second > 0


def test_search_invalid_engine():
    config=build_config(); cfg=StrategySearchConfig(candidate_count=2, simulations_per_strategy=2)
    candidates=generate_search_candidates(config,cfg)
    with pytest.raises(ValueError):
        search_cpu(config,candidates,2,0,engine='gpu')
