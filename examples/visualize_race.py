"""Create a basic matplotlib race visualization."""

import matplotlib.pyplot as plt
import numpy as np

from basic_race import build_config
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1sim import Strategy, TireCompound, simulate_race


if __name__ == "__main__":
    config = build_config()
    result = simulate_race(config, Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD]), seed=42)
    laps = np.arange(1, config.track.number_of_laps + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    for driver_index, driver_result in enumerate(result.driver_results):
        positions = [row.index(driver_index) + 1 for row in result.positions_by_lap]
        axes[0].plot(laps, positions, label=driver_result.driver)
        axes[1].plot(np.arange(1, len(driver_result.lap_times) + 1), driver_result.lap_times, label=driver_result.driver)
        tire_age = []
        age = 0
        history = list(driver_result.tire_usage)
        for compound in history:
            next_age = config.track.number_of_laps if compound == history[-1] else age + 1
            tire_age.extend([compound.value] * max(1, next_age - age))
            age = next_age
        axes[2].step(np.arange(1, len(tire_age) + 1), tire_age, where="mid", label=driver_result.driver)

    axes[0].invert_yaxis()
    axes[0].set(title="Position vs Lap", xlabel="Lap", ylabel="Position")
    axes[1].set(title="Lap Time vs Lap", xlabel="Lap", ylabel="Seconds")
    axes[2].set(title="Tire Compound vs Lap", xlabel="Lap", ylabel="Compound")
    axes[0].legend()
    fig.savefig("race_visualization.png", dpi=140)
    print("Saved race_visualization.png")
