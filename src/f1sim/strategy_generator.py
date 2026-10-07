"""Generation of valid, realistic fixed race strategies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

from .config import SimulationConfig, TireCompound
from .models import Strategy
from .strategy import validate_strategy


@dataclass(frozen=True)
class StrategyGenerationConfig:
    """Controls the size and shape of generated strategy candidates."""

    min_stops: int = 1
    max_stops: int = 2
    min_segment_laps: int = 5
    max_strategies: int = 100
    dry_compounds_only: bool = True


def generate_strategies(config: SimulationConfig, generation: StrategyGenerationConfig | None = None) -> list[Strategy]:
    """Generate unique valid strategies without optimization."""
    generation = generation or StrategyGenerationConfig()
    if generation.min_stops < 0 or generation.max_stops < generation.min_stops:
        raise ValueError("Invalid stop range")
    if generation.min_segment_laps <= 0 or generation.max_strategies <= 0:
        raise ValueError("Segment length and strategy count must be positive")
    compounds = [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]
    if not generation.dry_compounds_only:
        compounds = list(TireCompound)
    generated: list[Strategy] = []
    seen: set[tuple] = set()
    for stops in range(generation.min_stops, generation.max_stops + 1):
        for sequence in permutations(compounds, stops + 1):
            if any(sequence[i] == sequence[i + 1] for i in range(len(sequence) - 1)):
                continue
            if stops == 0:
                pit_choices = [()]
            else:
                earliest = generation.min_segment_laps
                latest = config.track.number_of_laps - generation.min_segment_laps
                if latest <= earliest:
                    continue
                pit_choices = combinations(range(earliest, latest + 1), stops)
            for pit_laps in pit_choices:
                pit_laps = tuple(pit_laps)
                if stops and any((b - a) < generation.min_segment_laps for a, b in zip((0, *pit_laps), (*pit_laps, config.track.number_of_laps))):
                    continue
                strategy = Strategy(sequence[0], list(pit_laps), list(sequence[1:]))
                validate_strategy(strategy, config)
                key = (strategy.initial_compound, tuple(strategy.pit_laps), tuple(strategy.compounds))
                if key not in seen:
                    seen.add(key)
                    generated.append(strategy)
                    if len(generated) >= generation.max_strategies:
                        return generated
    return generated
