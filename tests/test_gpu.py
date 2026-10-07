"""GPU correctness and availability tests.

Run these against a real GPU with normal pytest. For environments without CUDA,
run the same tests with NUMBA_ENABLE_CUDASIM=1 to validate kernel semantics.
"""
from dataclasses import replace

import numpy as np
import pytest

from examples.basic_race import build_config
from f1sim import Strategy, TireCompound, simulate_race
from f1sim.gpu import cuda_available, simulate_strategy_gpu


pytestmark = [
    pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable"),
    pytest.mark.filterwarnings("ignore:overflow encountered in scalar:RuntimeWarning"),
]


def strategy() -> Strategy:
    return Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD])


def test_gpu_is_deterministic_for_fixed_seed() -> None:
    config = build_config()
    first = simulate_strategy_gpu(config, strategy(), 8, seed=42)
    second = simulate_strategy_gpu(config, strategy(), 8, seed=42)
    np.testing.assert_array_equal(first.race_times, second.race_times)
    np.testing.assert_array_equal(first.finishing_positions, second.finishing_positions)
    np.testing.assert_array_equal(first.pit_stops, second.pit_stops)


def test_gpu_matches_cpu_without_stochasticity() -> None:
    config = build_config()
    config = replace(
        config,
        stochastic_noise_std=0.0,
        dnf_base_probability=0.0,
        weather=replace(config.weather, transition_probability=0.0, safety_car_probability=0.0),
    )
    focal = strategy()
    cpu = simulate_race(config, [focal] * len(config.drivers), seed=42)
    gpu = simulate_strategy_gpu(config, focal, 1, seed=42)

    assert gpu.race_times[0] == pytest.approx(cpu.total_race_time, abs=1e-10)
    assert gpu.finishing_positions[0].tolist() == [r.finishing_position for r in cpu.driver_results]
    assert gpu.pit_stops[0].tolist() == [r.pit_stops for r in cpu.driver_results]


def test_gpu_has_independent_simulation_streams() -> None:
    result = simulate_strategy_gpu(build_config(), strategy(), 16, seed=100)
    assert len(np.unique(result.race_times)) > 1
    assert result.seeds.tolist() == list(range(100, 116))

@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_gpu_context_reduces_static_transfer_payload() -> None:
    from f1sim.gpu import create_gpu_context

    config = build_config()
    result = simulate_strategy_gpu(config, strategy(), 8, seed=42, return_timing=True)
    context = create_gpu_context(config)
    cached = simulate_strategy_gpu(config, strategy(), 8, seed=42, return_timing=True, context=context)
    assert cached.timing.host_to_device_bytes < result.timing.host_to_device_bytes
    np.testing.assert_array_equal(cached.race_times, result.race_times)


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_gpu_timing_reports_kernel_and_transfer_components() -> None:
    result = simulate_strategy_gpu(build_config(), strategy(), 8, seed=42, return_timing=True)
    assert result.timing.kernel_seconds >= 0
    assert result.timing.host_to_device_seconds >= 0
    assert result.timing.device_to_host_seconds >= 0
    assert result.timing.host_to_device_bytes > 0
    assert result.timing.device_to_host_bytes > 0
