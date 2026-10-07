"""Simple transparent performance/cost model for AI compute workloads."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUCostProfile:
    name: str
    dollars_per_hour: float
    power_watts: float | None = None


@dataclass(frozen=True)
class CostEstimate:
    runtime_seconds: float
    dollars: float
    energy_kwh: float | None
    simulations: int
    simulations_per_dollar: float


def estimate_cost(runtime_seconds: float, simulations: int, profile: GPUCostProfile) -> CostEstimate:
    if runtime_seconds < 0 or simulations <= 0 or profile.dollars_per_hour < 0:
        raise ValueError("runtime, simulations, and cost must be valid")
    dollars = runtime_seconds / 3600.0 * profile.dollars_per_hour
    energy = None if profile.power_watts is None else runtime_seconds / 3600.0 * profile.power_watts / 1000.0
    return CostEstimate(runtime_seconds, dollars, energy, simulations, simulations / dollars if dollars > 0 else float("inf"))
