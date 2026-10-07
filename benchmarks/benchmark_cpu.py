"""CPU baseline benchmark."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.basic_race import build_config
from f1sim import Strategy, TireCompound, simulate_race


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--races", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.races <= 0:
        raise SystemExit("--races must be positive")

    config = build_config()
    strategy = Strategy(TireCompound.MEDIUM, [15], [TireCompound.HARD])
    start = time.perf_counter()
    for offset in range(args.races):
        simulate_race(config, strategy, seed=args.seed + offset)
    elapsed = time.perf_counter() - start
    print(f"Races: {args.races}")
    print(f"Total time: {elapsed:.6f} seconds")
    print(f"Average race: {elapsed / args.races * 1000:.3f} ms")
    print(f"Races/sec: {args.races / elapsed:.2f}")


if __name__ == "__main__":
    main()
