"""Small GPU-aware AI workload scheduler used by Phase 12."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from typing import Any, Callable


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass(frozen=True)
class GPUResource:
    gpu_id: int
    memory_gb: float = 0.0
    tags: tuple[str, ...] = ()


@dataclass
class GPUJob:
    job_id: str
    workload: str
    priority: int
    required_memory_gb: float = 0.0
    batch_size: int = 1
    retries: int = 0
    state: JobState = JobState.QUEUED
    assigned_gpu: int | None = None
    result: Any = None
    error: str | None = None


class GPUScheduler:
    """Priority scheduler with explicit allocation and retry state."""
    def __init__(self, resources: list[GPUResource]):
        if not resources:
            raise ValueError("at least one GPU resource is required")
        self.resources = {r.gpu_id: r for r in resources}
        self.available = set(self.resources)
        self.jobs: dict[str, GPUJob] = {}
        self._queue: list[tuple[int, int, str]] = []
        self._sequence = count()

    def submit(self, job: GPUJob) -> None:
        if job.job_id in self.jobs:
            raise ValueError(f"duplicate job id: {job.job_id}")
        self.jobs[job.job_id] = job
        heappush(self._queue, (-job.priority, next(self._sequence), job.job_id))

    def next_job(self) -> tuple[GPUJob, GPUResource] | None:
        while self._queue and self.available:
            _, _, job_id = heappop(self._queue)
            job = self.jobs[job_id]
            candidates = [self.resources[g] for g in self.available if self.resources[g].memory_gb >= job.required_memory_gb]
            if not candidates:
                heappush(self._queue, (_, next(self._sequence), job_id))
                return None
            gpu = min(candidates, key=lambda x: x.gpu_id)
            self.available.remove(gpu.gpu_id)
            job.state = JobState.RUNNING
            job.assigned_gpu = gpu.gpu_id
            return job, gpu
        return None

    def complete(self, job_id: str, result: Any) -> None:
        job = self.jobs[job_id]
        job.state = JobState.COMPLETED
        job.result = result
        if job.assigned_gpu is not None:
            self.available.add(job.assigned_gpu)

    def fail(self, job_id: str, error: Exception | str, retry: bool = True) -> None:
        job = self.jobs[job_id]
        if job.assigned_gpu is not None:
            self.available.add(job.assigned_gpu)
        job.error = str(error)
        if retry and job.retries > 0:
            job.retries -= 1
            job.state = JobState.RETRYING
            job.state = JobState.QUEUED
            heappush(self._queue, (-job.priority, next(self._sequence), job.job_id))
        else:
            job.state = JobState.FAILED

    def utilization(self) -> dict[int, float]:
        return {gpu: (0.0 if gpu in self.available else 1.0) for gpu in self.resources}
