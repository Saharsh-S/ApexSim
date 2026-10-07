"""High-level simulator helpers."""

from __future__ import annotations

from .config import SimulationConfig
from .models import RaceResult, Strategy
from .race import simulate_race as _simulate_race


def simulate_race(config: SimulationConfig, strategy: Strategy | list[Strategy], seed: int = 0) -> RaceResult:
    """Run a single race using one strategy for all drivers or per-driver strategies."""
    strategies = strategy if isinstance(strategy, list) else [strategy] * len(config.drivers)
    return _simulate_race(config, strategies, seed)
