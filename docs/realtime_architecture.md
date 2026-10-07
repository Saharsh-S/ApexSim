# Real-Time Strategy Architecture

Phase 9 turns the simulator and ML components into a decision pipeline.

```text
race state
   |
   v
state validation / normalization
   |
   v
candidate generation
   |
   +------------------------------+
   |                              |
   v                              v
ML surrogate                  Monte Carlo
   |                         Python / NumPy / CUDA*
   +--------------+---------------+
                  |
                  v
             strategy ranking
                  |
                  v
               decision

* CUDA is selected only when an actual CUDA device is available.
```

## State

`RaceState` contains the current lap, position, tire, tire age, gaps, rain probability, safety-car probability, fuel, weather, and degradation.

## Candidate generation

The engine reuses Phase 6 deterministic strategy generation. When rain probability is material, the real-time layer also injects Intermediate and Wet candidates so that weather-dependent decisions are representable.

## Evaluation modes

### Simulation

Uses the existing NumPy Monte Carlo engine to score the current candidate set. This is slower but remains close to the simulator correctness reference.

### Surrogate

Uses the Phase 7 XGBoost models to score a candidate batch without running Monte Carlo simulations. This is intended as the low-latency path once a properly state-conditioned model is trained.

## Ranking

Candidates are ranked using predicted win/podium probability, expected race time, expected finishing position, and risk. The top candidate is returned with a confidence value derived from the score margin to the runner-up.

The confidence is a decision heuristic, not a calibrated probability of recommendation correctness.

## Latency budget

The default engineering target is 100 ms warm decision latency. The benchmark records:

- state processing;
- candidate generation;
- evaluation;
- ranking;
- total decision latency.

The target is a measurement goal, not a production claim.
