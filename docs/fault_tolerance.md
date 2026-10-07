# Fault Tolerance (Phase 13)

Checkpoints record completed simulation index ranges and failed workers. Recovery only resubmits missing ranges. Each simulation has a deterministic seed derived from `(global_seed, simulation_index)`, so moving a simulation to another worker does not change its random stream.

The production GPU implementation should persist checkpoints to durable storage and use heartbeats/timeouts. The included reference implementation focuses on deterministic recovery semantics.
