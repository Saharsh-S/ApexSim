"""Run a real-time strategy decision from a live-like race state."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.basic_race import build_config
from f1sim.config import TireCompound, WeatherType
from f1sim.realtime import RaceState, RealtimeStrategyEngine


def main() -> None:
    config = build_config()
    state = RaceState(
        lap=37, position=4, tire=TireCompound.MEDIUM, tire_age=18,
        gap_ahead=2.4, gap_behind=1.8, rain_probability=0.62,
        safety_car_probability=0.15, fuel=0.42, weather=WeatherType.DRY,
        tire_degradation=config.tires[TireCompound.MEDIUM].degradation_rate,
    )
    engine = RealtimeStrategyEngine(config, seed=42)
    decision = engine.decide(state, candidate_count=32, top_k=3, backend="simulation", simulations_per_candidate=8)
    print(f"Recommended strategy: {decision.recommendation.strategy}")
    print(f"Confidence: {decision.recommendation.confidence * 100:.1f}%")
    print(f"Expected finish: P{decision.recommendation.expected_finish:.1f}")
    print(f"Decision latency: {decision.timing.total_ms:.3f} ms")
    print("Alternatives:")
    for alt in decision.alternatives:
        print(f"  {alt.strategy} | P{alt.expected_finish:.1f} | confidence {alt.confidence * 100:.1f}%")


if __name__ == "__main__":
    main()
