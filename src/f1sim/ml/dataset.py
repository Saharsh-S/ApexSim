"""Synthetic strategy-outcome dataset generation."""
from __future__ import annotations
from dataclasses import dataclass
import csv, time
from pathlib import Path
import numpy as np
from ..config import SimulationConfig, TireCompound, WeatherType
from ..strategy_search import StrategySearchConfig, generate_search_candidates
from ..monte_carlo import simulate_strategies_numpy

FEATURES = ["lap","tire_age","tire_compound","weather","fuel","current_position","gap","safety_car","tire_degradation","start_tire","pit_lap","second_tire","second_pit","target_pace","risk_level"]
TARGETS = ["expected_race_time","finishing_position","podium_probability","win_probability"]

@dataclass(frozen=True)
class DatasetConfig:
    rows: int = 1000
    simulations_per_row: int = 20
    seed: int = 7

def generate_dataset(config: SimulationConfig, out_path: str | Path, dataset: DatasetConfig = DatasetConfig()) -> Path:
    rng = np.random.default_rng(dataset.seed)
    candidates = generate_search_candidates(config, StrategySearchConfig(candidate_count=dataset.rows, simulations_per_strategy=dataset.simulations_per_row, seed=dataset.seed))
    rows=[]
    for i, cand in enumerate(candidates):
        mc = simulate_strategies_numpy(config, [cand.strategy], dataset.simulations_per_row, dataset.seed + i*1009)[0]
        s=mc.statistics
        strategy=cand.strategy
        first = strategy.initial_compound.value
        second = strategy.compounds[0].value if strategy.compounds else first
        pit1 = strategy.pit_laps[0] if strategy.pit_laps else config.track.number_of_laps
        second_pit = strategy.pit_laps[1] if len(strategy.pit_laps)>1 else config.track.number_of_laps
        row={
          "lap": int(rng.integers(1, config.track.number_of_laps+1)),
          "tire_age": int(rng.integers(0, min(40, config.track.number_of_laps)+1)),
          "tire_compound": first, "weather": rng.choice([w.value for w in WeatherType]),
          "fuel": float(rng.uniform(.05,1.0)), "current_position": int(rng.integers(1,len(config.drivers)+1)),
          "gap": float(rng.uniform(0,8)), "safety_car": int(rng.random()<.12),
          "tire_degradation": float(config.tires[strategy.initial_compound].degradation_rate),
          "start_tire": first, "pit_lap": int(pit1), "second_tire": second, "second_pit": int(second_pit),
          "target_pace": cand.target_pace, "risk_level": cand.risk_level,
          "expected_race_time": s.mean_race_time, "finishing_position": 1.0,
          "podium_probability": s.podium_probability, "win_probability": s.win_probability,
        }
        pos=[]
        for race in mc.races:
            p=race.finishing_position[0] if race.finishing_position else None
            if p is not None: pos.append(p)
        row["finishing_position"] = float(np.mean(pos)) if pos else float(len(config.drivers)+1)
        rows.append(row)
    out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FEATURES+TARGETS); w.writeheader(); w.writerows(rows)
    return out

def load_dataset(path: str | Path):
    import pandas as pd
    return pd.read_csv(path)
