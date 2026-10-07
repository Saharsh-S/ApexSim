# F1 Strategy Lab Web App

The web layer is a thin public interface over the existing `f1sim` research package.

## Local development

From the repository root:

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000.

The API documentation is available at http://127.0.0.1:8000/docs.

## Production-style local run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The deployment intentionally falls back to CPU/NumPy when CUDA is unavailable. GPU results are never fabricated.
