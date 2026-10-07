# Simulated AI Compute Cluster (Phase 12)

`GPUScheduler` models a small GPU cluster with explicit resources, priority queues, memory requirements, allocation, completion, failure, and retry states. `run_cluster` provides a synchronous reference scheduler suitable for deterministic tests.

The abstraction is intentionally workload-agnostic: simulation, inference, or training jobs can share the same queue.
