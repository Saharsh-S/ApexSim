"""Random-number helpers used to keep stochastic behavior explicit."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create the simulator's deterministic random-number generator."""
    return np.random.default_rng(seed)
