"""Checkpoint and recovery primitives for distributed simulation jobs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Checkpoint:
    job_id: str
    completed_ranges: tuple[tuple[int, int], ...]
    failed_workers: tuple[int, ...] = ()
    metadata: dict[str, Any] | None = None

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "job_id": self.job_id,
            "completed_ranges": [list(x) for x in self.completed_ranges],
            "failed_workers": list(self.failed_workers),
            "metadata": self.metadata or {},
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        data = json.loads(Path(path).read_text())
        return cls(data["job_id"], tuple(tuple(x) for x in data["completed_ranges"]), tuple(data.get("failed_workers", [])), data.get("metadata", {}))

    def missing_ranges(self, total: int) -> tuple[tuple[int, int], ...]:
        covered = [False] * total
        for start, end in self.completed_ranges:
            for i in range(max(0, start), min(total, end)):
                covered[i] = True
        ranges = []
        start = None
        for i, done in enumerate(covered + [True]):
            if not done and start is None:
                start = i
            elif done and start is not None:
                ranges.append((start, i))
                start = None
        return tuple(ranges)


def deterministic_seed(global_seed: int, simulation_index: int) -> int:
    """Stable per-simulation seed, independent of worker placement."""
    value = (global_seed ^ (simulation_index * 0x9E3779B1)) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 27
    return int(value & 0x7FFFFFFF)
