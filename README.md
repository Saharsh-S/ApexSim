# F1 GPU Race Simulator / Strategy Lab

A reproducible ML Systems research project for studying how simulation parallelism and optimized ML inference affect real-time Formula 1 strategy search.

The repository contains the complete progression from a deterministic CPU simulator to CUDA/Triton kernels, ML surrogate inference, real-time strategy decisions, distributed simulation, GPU scheduling, fault tolerance, cost modeling, and a deployable web application.

## Public web application

The project now has a browser interface backed by FastAPI.

```text
Browser
   │
   ▼
FastAPI web application
   │
   ├── Monte Carlo simulator
   ├── real-time strategy engine
   ├── ML surrogate interface
   └── benchmark/system endpoints
           │
           ▼
     f1sim research core
           │
     ┌─────┼──────────────┐
     ▼     ▼              ▼
   Python NumPy       CUDA/Triton
                       when available
```

The frontend is intentionally a thin layer over the research implementation. The simulator does not get reimplemented in JavaScript.

## Quick start

Requirements: Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_web.py
```

Then open **http://127.0.0.1:8000**.

API documentation: **http://127.0.0.1:8000/docs**.

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

### Docker

```bash
docker compose up --build
```

Then open **http://127.0.0.1:8000**.

### VS Code

Do **not** use **Go Live / Live Server** for the application. The site needs the Python API. Run:

```bash
python scripts/run_web.py
```

## What the site does

### Real-time strategy engine

Enter:

- current lap
- race position
- tyre compound
- tyre age
- gap ahead/behind
- rain probability
- safety-car probability
- candidate count

The backend returns:

- recommended strategy
- confidence
- expected finishing position
- win probability
- podium probability
- alternatives
- measured decision latency

### Monte Carlo simulator

Run repeated races for a selected tyre/pit strategy and inspect:

- win probability
- podium probability
- mean race time
- DNF probability

### Performance page

The application surfaces benchmark artifacts from the reproducible study. Missing GPU hardware is reported as unavailable rather than replaced with estimated values.

## Reproducible benchmark suite

Run the complete benchmark pipeline:

```bash
python scripts/run_all_benchmarks.py
```

Run the distributed study:

```bash
python scripts/run_distributed_study.py
```

Results are written to `results/`.

## Research question

> How much can massively parallel GPU simulation and optimized ML inference accelerate real-time F1 strategy search?

The study compares Python CPU, NumPy CPU, CUDA baseline, optimized CUDA, Triton, PyTorch inference, optimized inference, distributed workers, scheduling, fault tolerance, and cost/energy models where measurements are available.

## Measurement policy

This project does not fabricate unavailable hardware measurements.

If an environment has no NVIDIA GPU, the benchmark records CUDA/Triton measurements as unavailable and can still execute CPU reference and fallback experiments. CPU CUDA-simulator measurements are explicitly labeled as simulator measurements.

## Project layout

```text
api/                 FastAPI public API
web/static/          Browser frontend
src/f1sim/           Research and simulation core
benchmarks/          Performance experiments
results/             Reproducible benchmark artifacts
tests/               Automated tests
examples/             Research examples
scripts/              Reproduction and web startup scripts
docs/                 Technical reports
ml/checkpoints/       Small model checkpoints
Dockerfile            Container deployment
docker-compose.yml    Local container orchestration
DEPLOYMENT.md         Local/public deployment instructions
FINAL_REPORT.md       Full project report
```

## Technical stack

- Python
- NumPy
- FastAPI
- Uvicorn
- Numba CUDA
- Triton (optional)
- PyTorch (optional ML stack)
- XGBoost (optional ML stack)
- Docker
- HTML/CSS/JavaScript frontend

## Research progression

1. CPU reference simulator
2. Monte Carlo engine
3. CUDA acceleration
4. CUDA performance engineering
5. Triton implementation
6. GPU strategy search
7. ML surrogate model
8. inference optimization
9. real-time strategy engine
10. reproducible ML Systems performance study
11. distributed multi-worker simulation
12. simulated AI compute cluster scheduler
13. fault tolerance and recovery
14. cost-aware infrastructure
15. final AI Systems benchmark
16. deployable public web platform

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for local, Docker, public container deployment, GPU deployment, and production-hardening guidance.

## Status

The public web layer is runnable on CPU-only systems. CUDA/Triton functionality remains hardware-dependent.

The research repository is designed to remain useful when GPU hardware is unavailable: CPU reference paths, tests, simulator validation, benchmark methodology, and orchestration code can still be reproduced.
