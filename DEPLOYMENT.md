# F1 Strategy Lab — Run and Deploy

The project now includes a browser frontend and FastAPI backend. The frontend is served by FastAPI, so there is no separate Node.js build step.

## 1. Local setup

Requirements:

- Python 3.11+
- pip
- Optional: NVIDIA CUDA + compatible drivers for GPU execution

From the repository root:

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the website:

```bash
python scripts/run_web.py
```

Open:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

### Important

Do **not** use VS Code Live Server / Go Live for this application. Live Server only serves static files and would bypass the Python API. The correct development server is Uvicorn through `scripts/run_web.py`.

## 2. Docker

If Docker is installed:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000
```

Or run directly:

```bash
docker build -t f1-strategy-lab .
docker run --rm -p 8000:8000 f1-strategy-lab
```

## 3. Public deployment

The included `Dockerfile` is designed for container platforms. A simple deployment pattern is:

1. Push the repository to GitHub.
2. Create a new web service on a container-capable hosting provider.
3. Point it at the repository.
4. Select Dockerfile deployment.
5. Expose port `8000`.
6. Set `F1_CORS_ORIGINS` to your public origin instead of `*` when the deployment is stable.
7. Deploy.

The service starts with:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

For a first public CPU deployment, no GPU is required. The API reports CUDA/Triton availability and uses the CPU/NumPy path when GPU hardware is unavailable.

## 4. GPU deployment

A GPU host must have:

- NVIDIA driver
- CUDA-compatible runtime
- a compatible Numba/CUDA environment
- any optional Triton dependencies required by the benchmark

Install the GPU extras where appropriate:

```bash
pip install -e '.[gpu]'
```

Triton is intentionally optional:

```bash
pip install -e '.[triton]'
```

The website does not claim GPU acceleration unless the runtime actually detects it.

## 5. Public-server considerations

This repository is a research/demo deployment rather than a hardened multi-tenant SaaS product. Before exposing it to untrusted traffic at scale, add:

- authentication if private workloads are required
- a persistent job queue for long simulations
- per-user rate limiting
- request timeouts
- worker isolation
- persistent job storage
- HTTPS through the hosting provider
- monitoring and structured logs
- GPU scheduling quotas

The current API applies conservative request bounds so a normal public demo cannot request arbitrarily large synchronous simulations.
