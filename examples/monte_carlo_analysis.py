"""Run strategy Monte Carlo analysis and save the Phase 2 plots."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.basic_race import build_config
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1sim import StrategyGenerationConfig, generate_strategies
from f1sim.analysis import (
    plot_convergence, plot_race_time_distribution, plot_risk_reward,
    plot_strategy_win_probability, rank_strategies,
)
from f1sim.monte_carlo import simulate_strategies_numpy


def main() -> None:
    config = build_config()
    strategies = generate_strategies(config, StrategyGenerationConfig(max_strategies=4, min_stops=1, max_stops=2))
    results = simulate_strategies_numpy(config, strategies, 1000, seed=42)
    for stat in rank_strategies(results, by="win_probability"):
        print(f"{stat.strategy_id:42s} Win %={stat.win_probability * 100:6.2f}  Avg Time={stat.mean_race_time:8.2f}")

    output = os.path.join(ROOT, "phase2_plots")
    os.makedirs(output, exist_ok=True)
    plot_strategy_win_probability(results, os.path.join(output, "strategy_win_probability.png"))
    plot_race_time_distribution(results[0], os.path.join(output, "race_time_distribution.png"))
    plot_risk_reward(results, os.path.join(output, "strategy_risk_reward.png"))
    plot_convergence(results[0], os.path.join(output, "monte_carlo_convergence.png"))
    print(f"Plots written to {output}")


if __name__ == "__main__":
    main()
