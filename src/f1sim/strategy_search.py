"""Automatic candidate generation and CPU/GPU strategy search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from time import perf_counter
from typing import Sequence

import numpy as np

from .config import SimulationConfig, TireCompound
from .models import Strategy
from .monte_carlo import MonteCarloResult, simulate_strategies_numpy, simulate_strategies_python, strategy_id
from .strategy import validate_strategy
from .strategy_generator import StrategyGenerationConfig
from .gpu import cuda_available, simulate_strategy_gpu


@dataclass(frozen=True)
class StrategySearchConfig:
    """Controls automatic strategy-space sampling."""

    candidate_count: int = 1000
    simulations_per_strategy: int = 100
    min_stops: int = 1
    max_stops: int = 2
    min_segment_laps: int = 5
    seed: int = 0
    target_pace_min: float = 0.96
    target_pace_max: float = 1.02
    risk_min: float = 0.0
    risk_max: float = 1.0

    def __post_init__(self) -> None:
        if self.candidate_count <= 0 or self.simulations_per_strategy <= 0:
            raise ValueError("candidate_count and simulations_per_strategy must be positive")
        if self.min_stops < 0 or self.max_stops < self.min_stops:
            raise ValueError("invalid stop range")
        if self.min_segment_laps <= 0:
            raise ValueError("min_segment_laps must be positive")
        if self.target_pace_min <= 0 or self.target_pace_max < self.target_pace_min:
            raise ValueError("invalid target pace range")
        if not 0 <= self.risk_min <= self.risk_max <= 1:
            raise ValueError("risk range must be within [0, 1]")


@dataclass(frozen=True)
class SearchCandidate:
    """A strategy plus search-only pace/risk controls."""

    candidate_id: int
    strategy: Strategy
    target_pace: float
    risk_level: float

    @property
    def strategy_id(self) -> str:
        return f"{strategy_id(self.strategy)}|pace={self.target_pace:.4f}|risk={self.risk_level:.3f}"


@dataclass(frozen=True)
class StrategySearchResult:
    """Search output and timing information."""

    candidates: tuple[SearchCandidate, ...]
    results: tuple[MonteCarloResult, ...]
    top_indices: tuple[int, ...]
    backend: str
    candidate_count: int
    simulations_per_strategy: int
    total_simulations: int
    runtime_seconds: float
    strategies_per_second: float
    simulations_per_second: float

    @property
    def top_results(self) -> tuple[MonteCarloResult, ...]:
        return tuple(self.results[i] for i in self.top_indices)


def _valid_strategy_pool(config: SimulationConfig, generation: StrategySearchConfig) -> list[Strategy]:
    """Enumerate valid strategy skeletons without manually listing strategies."""
    compounds = (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)
    pool: list[Strategy] = []
    seen: set[tuple] = set()
    for stops in range(generation.min_stops, generation.max_stops + 1):
        for sequence in permutations(compounds, stops + 1):
            if any(a == b for a, b in zip(sequence, sequence[1:])):
                continue
            if stops == 0:
                pit_choices = [()]
            else:
                lo = generation.min_segment_laps
                hi = config.track.number_of_laps - generation.min_segment_laps
                pit_choices = combinations(range(lo, hi + 1), stops) if hi >= lo else []
            for pits in pit_choices:
                if any(b - a < generation.min_segment_laps for a, b in zip((0, *pits), (*pits, config.track.number_of_laps))):
                    continue
                candidate = Strategy(sequence[0], list(pits), list(sequence[1:]))
                validate_strategy(candidate, config)
                key = (candidate.initial_compound, tuple(candidate.pit_laps), tuple(candidate.compounds))
                if key not in seen:
                    seen.add(key)
                    pool.append(candidate)
    if not pool:
        raise ValueError("No valid strategies exist for the supplied race configuration")
    return pool


def generate_search_candidates(config: SimulationConfig, search: StrategySearchConfig) -> list[SearchCandidate]:
    """Sample a large, deterministic candidate space from valid strategy skeletons."""
    pool = _valid_strategy_pool(config, search)
    rng = np.random.default_rng(search.seed)
    candidates: list[SearchCandidate] = []
    # Candidate count is exact. Reusing a tire/pit skeleton is intentional:
    # target pace and risk are continuous search dimensions, so the full space
    # is much larger than the finite set of tire/pit combinations.
    for candidate_id in range(search.candidate_count):
        idx = int(rng.integers(0, len(pool)))
        pace = float(rng.uniform(search.target_pace_min, search.target_pace_max))
        risk = float(rng.uniform(search.risk_min, search.risk_max))
        candidates.append(SearchCandidate(candidate_id, pool[idx], pace, risk))
    return candidates


def _effective_strategy(candidate: SearchCandidate) -> Strategy:
    """Return the Phase 1 strategy used by the simulator.

    Target pace and risk are search dimensions. The current simulator models
    pace through the fixed driver/car parameters, so these controls are retained
    as metadata for ranking and future GPU strategy-conditioned kernels rather
    than silently changing the Phase 1 race equations.
    """
    return candidate.strategy


def _rank_indices(candidates: Sequence[SearchCandidate], results: Sequence[MonteCarloResult], top_k: int) -> tuple[int, ...]:
    if top_k <= 0:
        return ()
    scores = []
    for i, (candidate, result) in enumerate(zip(candidates, results)):
        # Prefer winning, then podiums, while penalizing variance and risk.
        stat = result.statistics
        score = (
            4.0 * stat.win_probability
            + 1.5 * stat.podium_probability
            - 0.002 * stat.risk_adjusted_score
            - 0.15 * candidate.risk_level
        )
        scores.append((score, i))
    scores.sort(reverse=True)
    return tuple(i for _, i in scores[:top_k])


def search_cpu(
    config: SimulationConfig,
    candidates: Sequence[SearchCandidate],
    simulations_per_strategy: int,
    seed: int = 0,
    engine: str = "numpy",
    top_k: int = 10,
) -> StrategySearchResult:
    """Evaluate candidates using the Python or NumPy CPU baseline."""
    if engine not in {"python", "numpy"}:
        raise ValueError("engine must be 'python' or 'numpy'")
    start = perf_counter()
    strategies = [_effective_strategy(c) for c in candidates]
    runner = simulate_strategies_python if engine == "python" else simulate_strategies_numpy
    results = runner(config, strategies, simulations_per_strategy, seed)
    elapsed = perf_counter() - start
    total = len(candidates) * simulations_per_strategy
    return StrategySearchResult(tuple(candidates), tuple(results), _rank_indices(candidates, results, top_k), engine,
                                len(candidates), simulations_per_strategy, total, elapsed,
                                len(candidates) / elapsed if elapsed else float("inf"),
                                total / elapsed if elapsed else float("inf"))


def search_gpu(
    config: SimulationConfig,
    candidates: Sequence[SearchCandidate],
    simulations_per_strategy: int,
    seed: int = 0,
    top_k: int = 10,
) -> StrategySearchResult:
    """Evaluate candidates on CUDA, using simulation-level GPU parallelism per candidate."""
    if not cuda_available():
        raise RuntimeError("CUDA device is required for GPU strategy search")
    start = perf_counter()
    results: list[MonteCarloResult] = []
    # The current Phase 3 kernel maps one thread to one race. Candidate-level
    # scheduling therefore remains on the host while all simulations for each
    # candidate execute concurrently on the GPU. This keeps the Phase 1/2
    # simulator as the reference rather than introducing a second race model.
    for i, candidate in enumerate(candidates):
        batch = simulate_strategy_gpu(config, _effective_strategy(candidate), simulations_per_strategy,
                                      seed + i * simulations_per_strategy)
        # Convert the compact GPU output into the common aggregate representation.
        races = []
        from .monte_carlo import MonteCarloRace, aggregate_statistics
        for j in range(simulations_per_strategy):
            races.append(MonteCarloRace(
                seed=int(batch.seeds[j]), winner=None,
                finishing_position=tuple(int(x) if int(x) > 0 else None for x in batch.finishing_positions[j]),
                total_race_time=float(batch.race_times[j]),
                fastest_lap=float(np.min(batch.fastest_laps[j])) if np.any(batch.fastest_laps[j] >= 0) else None,
                pit_stops=int(np.sum(batch.pit_stops[j])), tire_usage=tuple(),
                safety_car_laps=int(batch.safety_car_laps[j]), dnf=bool(batch.dnf[j]),
                total_fuel_used=float(len(config.drivers)), strategy_id=candidate.strategy_id,
            ))
        # GPU winner names are reconstructed from position 1.
        fixed = []
        for race in races:
            winner = None
            for idx, pos in enumerate(race.finishing_position):
                if pos == 1:
                    winner = config.drivers[idx].name
                    break
            fixed.append(MonteCarloRace(**{**race.__dict__, "winner": winner}))
        stats = aggregate_statistics(fixed, candidate.strategy_id)
        results.append(MonteCarloResult(candidate.strategy, candidate.strategy_id,
                                        tuple(int(x) for x in batch.seeds), tuple(fixed), stats))
    elapsed = perf_counter() - start
    total = len(candidates) * simulations_per_strategy
    return StrategySearchResult(tuple(candidates), tuple(results), _rank_indices(candidates, results, top_k), "cuda",
                                len(candidates), simulations_per_strategy, total, elapsed,
                                len(candidates) / elapsed if elapsed else float("inf"),
                                total / elapsed if elapsed else float("inf"))


@dataclass(frozen=True)
class StrategySearchSummary:
    """User-facing metrics for one ranked candidate."""

    rank: int
    candidate_id: int
    strategy_id: str
    expected_finishing_position: float
    win_probability: float
    podium_probability: float
    expected_race_time: float
    risk: float
    expected_tire_usage: float
    expected_pit_stops: float


def summarize_top_strategies(result: StrategySearchResult, top_k: int = 10) -> tuple[StrategySearchSummary, ...]:
    """Return the requested top-strategy metrics in ranking order."""
    summaries=[]
    for rank, idx in enumerate(result.top_indices[:top_k], 1):
        candidate=result.candidates[idx]; mc=result.results[idx]; stat=mc.statistics
        positions=[]
        for race in mc.races:
            p=race.finishing_position[0] if race.finishing_position else None
            positions.append(float(p if p is not None else len(race.finishing_position)+1))
        tire_usage=[len(race.tire_usage) for race in mc.races]
        summaries.append(StrategySearchSummary(
            rank=rank, candidate_id=candidate.candidate_id, strategy_id=candidate.strategy_id,
            expected_finishing_position=float(np.mean(positions)), win_probability=stat.win_probability,
            podium_probability=stat.podium_probability, expected_race_time=stat.mean_race_time,
            risk=candidate.risk_level, expected_tire_usage=float(candidate.strategy.number_of_stops + 1),
            expected_pit_stops=float(candidate.strategy.number_of_stops)))
    return tuple(summaries)
