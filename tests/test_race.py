from f1sim import Strategy, TireCompound, simulate_race
from examples.basic_race import build_config


def test_race_completes_expected_laps():
    config = build_config()
    result = simulate_race(config, Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD]), seed=7)
    assert len(result.weather_by_lap) == config.track.number_of_laps
    assert len(result.positions_by_lap) == config.track.number_of_laps


def test_pit_stop_changes_tire_and_adds_time():
    config = build_config()
    result = simulate_race(config, Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD]), seed=7)
    for driver in result.driver_results:
        assert driver.pit_stops == 1
        assert driver.total_pit_time == config.track.pit_lane_time_loss
        assert driver.tire_usage == (TireCompound.MEDIUM, TireCompound.HARD)


def test_driver_order_is_classified():
    config = build_config()
    result = simulate_race(config, Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD]), seed=7)
    positions = [d.finishing_position for d in result.driver_results if not d.dnf]
    assert positions == sorted(positions)
