"""Tests for the optional Triton hot-kernel backend."""

from __future__ import annotations

import numpy as np
import pytest

from f1sim.gpu.triton import compute_lap_times_cpu, compute_lap_times_triton, triton_available


def _state(races=3, drivers=4):
    rng = np.random.default_rng(7)
    total = rng.uniform(0, 8, (races, drivers)).astype(np.float32)
    fuel = np.full((races, drivers), 0.5, dtype=np.float32)
    age = np.tile(np.arange(drivers, dtype=np.int32), (races, 1))
    tire = np.mod(np.arange(races * drivers, dtype=np.int32).reshape(races, drivers), 5)
    ahead = np.full((races, drivers), -1, dtype=np.int32)
    for r in range(races):
        order = np.argsort(total[r])
        for p in range(1, drivers):
            ahead[r, order[p]] = order[p - 1]
    c = dict(
        driver_skill=np.array([.92,.85,.79,.73], dtype=np.float32)[:drivers],
        driver_aggression=np.array([.55,.5,.45,.4], dtype=np.float32)[:drivers],
        driver_management=np.array([.78,.72,.68,.62], dtype=np.float32)[:drivers],
        driver_consistency=np.ones(drivers, dtype=np.float32) * .9,
        car_performance=np.array([1.04,1,.97,.94], dtype=np.float32)[:drivers],
        car_tire_performance=np.ones(drivers, dtype=np.float32),
        tire_pace=np.array([.45,.25,.1,-.2,-.35], dtype=np.float32),
        tire_degradation=np.array([.018,.012,.009,.008,.007], dtype=np.float32),
        tire_warmup=np.array([2,2,3,1,2], dtype=np.float32),
        weather_mismatch=np.array([[0,0,0,9,15],[4,3,3.5,0,2],[11,10,10,2,0]], dtype=np.float32),
        weather_id=0, weather_base=0.0, base_lap_time=90.0, fuel_effect=.035,
        traffic_base=.6, overtaking_difficulty=.5, degradation_multiplier=1.0,
    )
    return dict(total=total, fuel=fuel, tire_age=age, tire_id=tire, ahead_index=ahead), c


def test_cpu_hot_slice_is_deterministic():
    state, c = _state()
    a = compute_lap_times_cpu(**state, **c)
    b = compute_lap_times_cpu(**state, **c)
    np.testing.assert_array_equal(a, b)
    assert np.all(a >= 1.0)


def test_cpu_hot_slice_edge_single_driver():
    state, c = _state(races=1, drivers=1)
    out = compute_lap_times_cpu(**state, **c)
    assert out.shape == (1, 1)
    assert np.isfinite(out).all()


@pytest.mark.skipif(not triton_available(), reason="Triton is not installed")
def test_triton_matches_cpu_hot_slice():
    state, c = _state(races=8)
    cpu = compute_lap_times_cpu(**state, **c)
    result = compute_lap_times_triton(**state, **c)
    np.testing.assert_allclose(result.lap_times, cpu, rtol=2e-4, atol=2e-3)


@pytest.mark.skipif(not triton_available(), reason="Triton is not installed")
def test_triton_block_size_validation():
    state, c = _state(drivers=4)
    with pytest.raises(ValueError):
        compute_lap_times_triton(**state, **c, block_size=2)
