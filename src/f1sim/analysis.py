"""Monte Carlo analysis, ranking, convergence, and visualization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .monte_carlo import MonteCarloResult, StrategyStatistics, strategy_id


def rank_strategies(results: Sequence[MonteCarloResult], by: str = "win_probability") -> list[StrategyStatistics]:
    """Rank strategies by a supported aggregate metric."""
    if not results:
        return []
    fields = {"win_probability", "mean_race_time", "podium_probability", "risk_adjusted_score"}
    if by not in fields:
        raise ValueError(f"Unsupported ranking metric: {by}")
    reverse = by in {"win_probability", "podium_probability"}
    return sorted((r.statistics for r in results), key=lambda x: getattr(x, by), reverse=reverse)


def convergence_curve(result: MonteCarloResult) -> tuple[np.ndarray, np.ndarray]:
    """Return sample counts and cumulative win probabilities."""
    wins = np.asarray([r.finishing_position[0] == 1 for r in result.races], dtype=float)
    counts = np.arange(1, len(wins) + 1)
    return counts, np.cumsum(wins) / counts


def strategy_risk_reward(results: Sequence[MonteCarloResult]) -> list[tuple[str, float, float]]:
    """Return strategy id, win probability, and risk-adjusted score."""
    return [(r.strategy_id, r.statistics.win_probability, r.statistics.risk_adjusted_score) for r in results]


def plot_strategy_win_probability(results: Sequence[MonteCarloResult], path: str | None = None):
    """Plot win probability by strategy."""
    import matplotlib.pyplot as plt
    ranked = rank_strategies(results, "win_probability")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([x.strategy_id for x in ranked], [x.win_probability for x in ranked])
    ax.set_ylabel("Win probability")
    ax.set_xlabel("Strategy")
    ax.set_title("Monte Carlo Strategy Win Probability")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_race_time_distribution(result: MonteCarloResult, path: str | None = None):
    """Plot race-time distribution for one strategy."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist([r.total_race_time for r in result.races], bins=30)
    ax.set_xlabel("Race time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Race Time Distribution: {result.strategy_id}")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_risk_reward(results: Sequence[MonteCarloResult], path: str | None = None):
    """Plot win probability against risk-adjusted race time."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter([r.statistics.risk_adjusted_score for r in results], [r.statistics.win_probability for r in results])
    for result in results:
        ax.annotate(result.strategy_id, (result.statistics.risk_adjusted_score, result.statistics.win_probability), fontsize=8)
    ax.set_xlabel("Risk-adjusted score (lower is better)")
    ax.set_ylabel("Win probability")
    ax.set_title("Strategy Risk / Reward")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_convergence(result: MonteCarloResult, path: str | None = None):
    """Plot estimated win probability as simulation count increases."""
    import matplotlib.pyplot as plt
    counts, probabilities = convergence_curve(result)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(counts, probabilities)
    ax.axhline(result.statistics.win_probability, linestyle="--", label="Final estimate")
    ax.set_xlabel("Simulations")
    ax.set_ylabel("Estimated win probability")
    ax.set_title(f"Monte Carlo Convergence: {result.strategy_id}")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
