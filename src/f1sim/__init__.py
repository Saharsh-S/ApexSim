"""F1 GPU Race Simulator: CPU reference and Monte Carlo baselines."""

from .config import (
    CarConfig, DriverConfig, SimulationConfig, TireCompound, TireConfig,
    TrackConfig, WeatherConfig, WeatherType,
)
from .models import DriverResult, RaceResult, RaceState, Strategy
from .simulator import simulate_race
from .monte_carlo import (
    BenchmarkResult, MonteCarloRace, MonteCarloResult, StrategyStatistics,
    simulate_strategy, simulate_strategies_numpy, simulate_strategies_python,
)
from .strategy_generator import StrategyGenerationConfig, generate_strategies
from .gpu import GPURaceBatch, GPUExecutionTiming, GPUDeviceContext, create_gpu_context, cuda_available, cuda_backend_status, simulate_strategy_gpu

__all__ = [
    "CarConfig", "DriverConfig", "SimulationConfig", "TireCompound", "TireConfig",
    "TrackConfig", "WeatherConfig", "WeatherType", "DriverResult", "RaceResult",
    "RaceState", "Strategy", "simulate_race", "MonteCarloRace", "MonteCarloResult",
    "StrategyStatistics", "BenchmarkResult", "simulate_strategy", "simulate_strategies_python",
    "simulate_strategies_numpy", "StrategyGenerationConfig", "generate_strategies",
    "GPURaceBatch", "GPUExecutionTiming", "GPUDeviceContext", "create_gpu_context",
    "cuda_available", "cuda_backend_status", "simulate_strategy_gpu",
    "SearchCandidate", "StrategySearchConfig", "StrategySearchResult", "StrategySearchSummary", "generate_search_candidates", "search_cpu", "search_gpu", "summarize_top_strategies",
]
from .strategy_search import SearchCandidate, StrategySearchConfig, StrategySearchResult, StrategySearchSummary, generate_search_candidates, search_cpu, search_gpu, summarize_top_strategies
