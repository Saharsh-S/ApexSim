from examples.basic_race import build_config
from f1sim.config import TireCompound
from f1sim.models import Strategy
from f1sim.distributed import distributed_simulate
from f1sim.scheduler import GPUJob, GPUResource, GPUScheduler, JobState
from f1sim.fault_tolerance import Checkpoint, deterministic_seed
from f1sim.cost_model import GPUCostProfile, estimate_cost


def test_distributed_numpy_partition_and_aggregation():
    result = distributed_simulate(build_config(), Strategy(TireCompound.MEDIUM,[20],[TireCompound.HARD]), 8, gpu_ids=[0,1], backend="numpy")
    assert result.simulations == 8
    assert result.active_workers == 2
    assert len(result.statistics.__dict__) > 0
    assert sum(w.simulations for w in result.worker_results) == 8


def test_scheduler_priority_and_retry():
    s=GPUScheduler([GPUResource(0,16),GPUResource(1,16)])
    s.submit(GPUJob("low","simulation",1, retries=1)); s.submit(GPUJob("high","inference",5, retries=1))
    job,gpu=s.next_job(); assert job.job_id=="high"
    s.fail(job.job_id,"boom",retry=True); assert s.jobs[job.job_id].state==JobState.QUEUED


def test_checkpoint_recovery_and_deterministic_seed():
    cp=Checkpoint("x",((0,3),(5,7)))
    assert cp.missing_ranges(8)==((3,5),(7,8))
    assert deterministic_seed(1,4)==deterministic_seed(1,4)


def test_cost_model():
    e=estimate_cost(3600,1_000_000,GPUCostProfile("g",2,200))
    assert e.dollars==2
    assert e.energy_kwh==0.2
    assert e.simulations_per_dollar==500_000
