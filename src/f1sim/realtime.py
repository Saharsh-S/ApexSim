"""Real-time race-strategy decision engine built on the existing search/ML stack."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np

from .config import SimulationConfig, TireCompound, WeatherType
from .models import Strategy
from .monte_carlo import simulate_strategies_numpy
from .strategy_search import SearchCandidate, StrategySearchConfig, generate_search_candidates


@dataclass(frozen=True)
class RaceState:
    lap: int
    position: int
    tire: TireCompound
    tire_age: int
    gap_ahead: float
    gap_behind: float
    rain_probability: float
    safety_car_probability: float
    fuel: float = 0.5
    weather: WeatherType = WeatherType.DRY
    tire_degradation: float = 0.0

    def __post_init__(self) -> None:
        if self.lap <= 0 or self.position <= 0 or self.tire_age < 0:
            raise ValueError("lap, position, and tire_age must be positive/non-negative")
        if self.gap_ahead < 0 or self.gap_behind < 0:
            raise ValueError("gaps cannot be negative")
        for value, name in ((self.rain_probability, "rain_probability"), (self.safety_car_probability, "safety_car_probability"), (self.fuel, "fuel")):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True)
class StrategyDecision:
    strategy: str
    confidence: float
    expected_finish: float
    expected_race_time: float
    podium_probability: float
    win_probability: float
    score: float
    rank: int


@dataclass(frozen=True)
class DecisionTiming:
    state_processing_ms: float
    candidate_generation_ms: float
    evaluation_ms: float
    ranking_ms: float
    total_ms: float


@dataclass(frozen=True)
class RealtimeDecision:
    recommendation: StrategyDecision
    alternatives: tuple[StrategyDecision, ...]
    candidates_evaluated: int
    backend: str
    timing: DecisionTiming


def _strategy_label(strategy: Strategy) -> str:
    parts = [strategy.initial_compound.value]
    for pit, compound in zip(strategy.pit_laps, strategy.compounds):
        parts.append(f"PIT@{pit}")
        parts.append(compound.value)
    return " → ".join(parts)


def _candidate_features(state: RaceState, candidates: Sequence[SearchCandidate], config: SimulationConfig):
    import pandas as pd
    rows = []
    for candidate in candidates:
        s = candidate.strategy
        second = s.compounds[0].value if s.compounds else s.initial_compound.value
        pit1 = s.pit_laps[0] if s.pit_laps else config.track.number_of_laps
        pit2 = s.pit_laps[1] if len(s.pit_laps) > 1 else config.track.number_of_laps
        rows.append({
            "lap": state.lap, "tire_age": state.tire_age,
            "tire_compound": state.tire.value, "weather": state.weather.value,
            "fuel": state.fuel, "current_position": state.position,
            "gap": state.gap_ahead, "safety_car": int(state.safety_car_probability >= 0.5),
            "tire_degradation": state.tire_degradation,
            "start_tire": s.initial_compound.value, "pit_lap": pit1,
            "second_tire": second, "second_pit": pit2,
            "target_pace": candidate.target_pace, "risk_level": candidate.risk_level,
        })
    return pd.DataFrame(rows)


def _surrogate_rank(candidates, predictions):
    time_pred = np.asarray(predictions["expected_race_time"], dtype=float)
    finish_pred = np.asarray(predictions["finishing_position"], dtype=float)
    podium = np.clip(np.asarray(predictions["podium_probability"], dtype=float), 0, 1)
    win = np.clip(np.asarray(predictions["win_probability"], dtype=float), 0, 1)
    risk = np.asarray([c.risk_level for c in candidates], dtype=float)
    # Normalize race time so the score is dimensionless and stable across tracks.
    tscale = max(float(np.ptp(time_pred)), 1e-9)
    tnorm = (time_pred - np.min(time_pred)) / tscale
    score = 2.0 * win + 1.0 * podium - 0.45 * tnorm - 0.15 * risk - 0.25 * (finish_pred - np.min(finish_pred)) / max(float(np.ptp(finish_pred)), 1e-9)
    order = np.argsort(-score, kind="stable")
    return order, score, time_pred, finish_pred, podium, win


class RealtimeStrategyEngine:
    """Warm real-time strategy recommender.

    ``backend='surrogate'`` performs cheap ML ranking. ``backend='simulation'``
    evaluates the generated candidates with the existing NumPy Monte Carlo engine.
    CUDA validation can be supplied by a caller through a future GPU adapter; no
    CUDA result is fabricated when hardware is absent.
    """

    def __init__(self, config: SimulationConfig, seed: int = 17):
        self.config = config
        self.seed = seed

    def decide(
        self,
        state: RaceState,
        *,
        candidate_count: int = 64,
        top_k: int = 3,
        backend: str = "simulation",
        simulations_per_candidate: int = 8,
        xgb_models=None,
    ) -> RealtimeDecision:
        if candidate_count <= 0 or top_k <= 0:
            raise ValueError("candidate_count and top_k must be positive")
        if backend not in {"simulation", "surrogate"}:
            raise ValueError("backend must be 'simulation' or 'surrogate'")

        t0 = perf_counter()
        # State validation + compact derived values happen before candidate generation.
        _ = state.lap / self.config.track.number_of_laps
        _ = state.rain_probability + state.safety_car_probability
        state_ms = (perf_counter() - t0) * 1000

        t1 = perf_counter()
        search_cfg = StrategySearchConfig(
            candidate_count=candidate_count,
            simulations_per_strategy=simulations_per_candidate,
            min_stops=0,
            max_stops=2,
            min_segment_laps=5,
            seed=self.seed + state.lap * 1009 + state.position,
        )
        candidates = generate_search_candidates(self.config, search_cfg)
        # Weather-aware additions are part of the real-time candidate space. The
        # generic Phase 6 generator intentionally focuses on dry compounds; the
        # decision engine must still be able to surface Intermediate/Wet choices.
        if state.rain_probability >= 0.35 and state.lap + 1 < self.config.track.number_of_laps:
            pit = min(self.config.track.number_of_laps - 1, max(state.lap + 1, state.lap + 2))
            wet_candidates = [
                SearchCandidate(candidate_count + 0, Strategy(state.tire, [pit], [TireCompound.INTERMEDIATE]), 0.98, 0.35),
                SearchCandidate(candidate_count + 1, Strategy(state.tire, [pit], [TireCompound.WET]), 1.00, 0.60),
            ]
            candidates = tuple(candidates[:-2] + wet_candidates) if len(candidates) >= 2 else tuple(candidates) + tuple(wet_candidates)
        candidate_ms = (perf_counter() - t1) * 1000

        t2 = perf_counter()
        if backend == "surrogate":
            if xgb_models is None:
                raise ValueError("xgb_models is required for surrogate backend")
            from .ml.models import predict_xgboost
            frame = _candidate_features(state, candidates, self.config)
            predictions = predict_xgboost(xgb_models, frame)
            order, score, times, finishes, podium, win = _surrogate_rank(candidates, predictions)
            eval_ms = (perf_counter() - t2) * 1000
        else:
            results = simulate_strategies_numpy(
                self.config, [c.strategy for c in candidates], simulations_per_candidate,
                seed=self.seed + state.lap * 1009 + state.position,
            )
            times = np.array([r.statistics.mean_race_time for r in results], dtype=float)
            finishes = np.array([np.mean([race.finishing_position[0] for race in r.races if race.finishing_position and race.finishing_position[0] is not None]) if r.races else len(self.config.drivers) + 1 for r in results], dtype=float)
            podium = np.array([r.statistics.podium_probability for r in results], dtype=float)
            win = np.array([r.statistics.win_probability for r in results], dtype=float)
            risk = np.array([c.risk_level for c in candidates], dtype=float)
            score = 2.0 * win + podium - 0.01 * (times - np.min(times)) - 0.15 * risk
            order = np.argsort(-score, kind="stable")
            eval_ms = (perf_counter() - t2) * 1000

        t3 = perf_counter()
        best = order[0]
        margin = float(score[best] - score[order[1]]) if len(order) > 1 else 0.0
        confidence = float(np.clip(0.5 + 0.5 * margin / (abs(float(score[best])) + 1e-9), 0.0, 1.0))
        decisions = []
        for rank, idx in enumerate(order[:top_k], start=1):
            decisions.append(StrategyDecision(
                strategy=_strategy_label(candidates[idx].strategy), confidence=confidence if rank == 1 else float(max(0.0, confidence - 0.08 * (rank - 1))),
                expected_finish=float(finishes[idx]), expected_race_time=float(times[idx]),
                podium_probability=float(podium[idx]), win_probability=float(win[idx]),
                score=float(score[idx]), rank=rank,
            ))
        ranking_ms = (perf_counter() - t3) * 1000
        total_ms = state_ms + candidate_ms + eval_ms + ranking_ms
        return RealtimeDecision(decisions[0], tuple(decisions[1:]), len(candidates), backend,
                                DecisionTiming(state_ms, candidate_ms, eval_ms, ranking_ms, total_ms))
