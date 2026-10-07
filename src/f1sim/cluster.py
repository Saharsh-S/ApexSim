"""Phase 12 cluster orchestration and workload accounting."""
from __future__ import annotations
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from .scheduler import GPUJob, GPUResource, GPUScheduler

@dataclass(frozen=True)
class ClusterRun:
    jobs: int
    completed: int
    failed: int
    runtime_seconds: float
    utilization: dict[int, float]
    assignments: tuple[tuple[str, int], ...] = ()


def run_cluster(jobs: list[GPUJob], resources: list[GPUResource], handlers: dict[str, Callable[[GPUJob, int], object]]) -> ClusterRun:
    """Dispatch jobs concurrently while respecting GPU allocation and retries."""
    scheduler = GPUScheduler(resources)
    for job in jobs:
        scheduler.submit(job)
    start = perf_counter()
    completed = failed = 0
    assignments: list[tuple[str, int]] = []
    futures = {}
    with ThreadPoolExecutor(max_workers=len(resources)) as executor:
        while True:
            while len(futures) < len(resources):
                item = scheduler.next_job()
                if item is None:
                    break
                job, gpu = item
                assignments.append((job.job_id, gpu.gpu_id))
                futures[executor.submit(handlers[job.workload], job, gpu.gpu_id)] = (job, gpu)
            if not futures:
                if all(j.state.value in {"completed", "failed"} for j in scheduler.jobs.values()):
                    break
                raise RuntimeError("scheduler stalled with queued jobs")
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                job, _gpu = futures.pop(future)
                try:
                    scheduler.complete(job.job_id, future.result())
                    completed += 1
                except Exception as exc:
                    scheduler.fail(job.job_id, exc, retry=True)
                    if scheduler.jobs[job.job_id].state.value == "failed":
                        failed += 1
    return ClusterRun(len(jobs), completed, failed, perf_counter()-start, scheduler.utilization(), tuple(assignments))
