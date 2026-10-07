from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1sim.config import (
    CarConfig, DriverConfig, SimulationConfig, TireCompound, TireConfig,
    TrackConfig, WeatherConfig, WeatherType,
)
from f1sim.models import Strategy
from f1sim.monte_carlo import simulate_strategy
from f1sim.realtime import RaceState as RealtimeRaceState, RealtimeStrategyEngine
from f1sim.simulator import simulate_race

app = FastAPI(
    title="F1 GPU Race Simulator API",
    version="1.0.0",
    description="Public API for the F1 strategy simulation and ML systems research platform.",
)

origins = os.getenv("F1_CORS_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

# Serve the public web application assets. Keeping this mount explicit makes the
# frontend work both locally and inside the production container.
app.mount("/static", StaticFiles(directory=ROOT / "web" / "static", check_dir=True), name="static")


def build_config() -> SimulationConfig:
    tires = {
        TireCompound.SOFT: TireConfig(TireCompound.SOFT, 1.2, 0.070, 1.0, 18),
        TireCompound.MEDIUM: TireConfig(TireCompound.MEDIUM, 0.6, 0.045, 1.0, 28),
        TireCompound.HARD: TireConfig(TireCompound.HARD, 0.1, 0.025, 2.0, 38),
        TireCompound.INTERMEDIATE: TireConfig(TireCompound.INTERMEDIATE, -0.4, 0.035, 1.0, 25),
        TireCompound.WET: TireConfig(TireCompound.WET, -0.2, 0.025, 1.0, 30),
    }
    drivers = [
        DriverConfig("Driver 1", 0.915, 0.890, 0.72, 0.790, 0.920),
        DriverConfig("Driver 2", 0.905, 0.880, 0.70, 0.775, 0.915),
        DriverConfig("Driver 3", 0.895, 0.870, 0.74, 0.760, 0.905),
        DriverConfig("Driver 4", 0.885, 0.860, 0.68, 0.750, 0.895),
    ]
    cars = [
        CarConfig("Team A", 0.960, 0.995, 1.000),
        CarConfig("Team B", 0.955, 0.994, 0.998),
        CarConfig("Team C", 0.950, 0.993, 0.996),
        CarConfig("Team D", 0.945, 0.992, 0.994),
    ]
    return SimulationConfig(
        TrackConfig("Reference Ring", 60, 5.2, 92.0, 22.0, 0.55),
        drivers, cars, tires, WeatherConfig(safety_car_probability=0.02),
    )

CONFIG = build_config()

class SimulateRequest(BaseModel):
    tire: Literal["soft", "medium", "hard", "intermediate", "wet"] = "medium"
    pit_lap: int = Field(15, ge=1, le=59)
    second_tire: Literal["soft", "medium", "hard", "intermediate", "wet"] = "hard"
    simulations: int = Field(20, ge=1, le=2000)
    seed: int = Field(42, ge=0, le=2_000_000_000)

class StrategyRequest(BaseModel):
    lap: int = Field(..., ge=1, le=59)
    position: int = Field(..., ge=1, le=4)
    tire: Literal["soft", "medium", "hard", "intermediate", "wet"] = "medium"
    tire_age: int = Field(..., ge=0, le=60)
    gap_ahead: float = Field(..., ge=0, le=120)
    gap_behind: float = Field(..., ge=0, le=120)
    rain_probability: float = Field(..., ge=0, le=1)
    safety_car_probability: float = Field(..., ge=0, le=1)
    fuel: float = Field(0.5, ge=0, le=1)
    backend: Literal["simulation", "surrogate"] = "simulation"
    candidate_count: int = Field(32, ge=4, le=128)
    simulations_per_candidate: int = Field(4, ge=1, le=32)


def enum_compound(value: str) -> TireCompound:
    return TireCompound(value)


def result_json(result):
    stats = result.statistics
    return {
        "strategy": result.strategy_id,
        "simulations": stats.simulations,
        "meanRaceTime": stats.mean_race_time,
        "medianRaceTime": stats.median_race_time,
        "stdRaceTime": stats.std_race_time,
        "winProbability": stats.win_probability,
        "podiumProbability": stats.podium_probability,
        "dnfProbability": stats.dnf_probability,
        "meanPitStops": stats.mean_pit_stops,
        "riskAdjustedScore": stats.risk_adjusted_score,
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "f1-gpu-race-simulator", "version": app.version}

@app.get("/api/config")
def config():
    return {
        "track": {"name": CONFIG.track.name, "laps": CONFIG.track.number_of_laps, "lapDistanceKm": CONFIG.track.lap_distance_km},
        "drivers": [d.name for d in CONFIG.drivers],
        "teams": [c.team for c in CONFIG.cars],
        "tires": [t.value for t in TireCompound],
        "hardware": {"cuda": _cuda_available(), "triton": _triton_available()},
    }

@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    if req.pit_lap >= CONFIG.track.number_of_laps:
        raise HTTPException(400, "Pit lap must be before the final lap")
    try:
        strategy = Strategy(enum_compound(req.tire), [req.pit_lap], [enum_compound(req.second_tire)])
        result = simulate_strategy(CONFIG, strategy, req.simulations, req.seed)
        return result_json(result)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post("/api/strategy")
def strategy(req: StrategyRequest):
    if req.lap >= CONFIG.track.number_of_laps:
        raise HTTPException(400, "Lap must be before the final lap for a strategy decision")
    state = RealtimeRaceState(
        lap=req.lap, position=req.position, tire=enum_compound(req.tire), tire_age=req.tire_age,
        gap_ahead=req.gap_ahead, gap_behind=req.gap_behind, rain_probability=req.rain_probability,
        safety_car_probability=req.safety_car_probability, fuel=req.fuel,
    )
    engine = RealtimeStrategyEngine(CONFIG, seed=17)
    actual_backend = req.backend
    fallback_reason = None
    xgb_models = None

    if req.backend == "surrogate":
        # The public app must remain usable even when optional ML runtime packages
        # or checkpoint dependencies are absent. Load the bundled XGBoost artifact
        # when possible; otherwise transparently fall back to Monte Carlo.
        checkpoint = ROOT / "ml" / "checkpoints" / "phase8_xgboost.joblib"
        try:
            import joblib
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            xgb_models = joblib.load(checkpoint)
            actual_backend = "surrogate"
        except Exception as exc:
            actual_backend = "simulation"
            fallback_reason = f"ML surrogate unavailable; used Monte Carlo simulation instead ({type(exc).__name__})."

    try:
        decision = engine.decide(
            state,
            candidate_count=req.candidate_count,
            top_k=3,
            backend=actual_backend,
            simulations_per_candidate=req.simulations_per_candidate,
            xgb_models=xgb_models,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "recommendation": _decision_json(decision.recommendation),
        "alternatives": [_decision_json(x) for x in decision.alternatives],
        "candidatesEvaluated": decision.candidates_evaluated,
        "backend": decision.backend,
        "requestedBackend": req.backend,
        "fallback": fallback_reason,
        "timing": decision.timing.__dict__,
    }

@app.get("/api/benchmarks")
def benchmarks():
    path = ROOT / "results" / "final_benchmark_table.json"
    distributed = ROOT / "results" / "phase11_distributed_scaling.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    dist = json.loads(distributed.read_text()) if distributed.exists() else {}
    return {"final": data, "distributed": dist}

@app.get("/api/system")
def system():
    return {
        "python": sys.version.split()[0],
        "cuda": _cuda_available(),
        "triton": _triton_available(),
        "mode": "GPU-enabled" if _cuda_available() else "CPU fallback",
        "researchRule": "Unavailable hardware is reported as unavailable; no GPU performance is fabricated.",
    }


def _decision_json(x):
    return {
        "rank": x.rank, "strategy": x.strategy, "confidence": x.confidence,
        "expectedFinish": x.expected_finish, "expectedRaceTime": x.expected_race_time,
        "podiumProbability": x.podium_probability, "winProbability": x.win_probability,
        "score": x.score,
    }

def _cuda_available():
    try:
        from f1sim.gpu import cuda_available
        return bool(cuda_available())
    except Exception:
        return False

def _triton_available():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web" / "static" / "index.html")
