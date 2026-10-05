# FastAPI backend

The API loads the trained sklearn pipeline and category metadata from the project `artifacts/` directory at startup. It intentionally fails to start if those artifacts are missing. Category options are served from training metadata, and unknown locations/types are rejected. CORS permits `http://localhost:3100` and `http://127.0.0.1:3100`.

From the project root, create/activate `backend/.venv` and install dependencies:

```sh
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

While inside `backend/`, start it with:

```sh
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8100
```

Health: `GET http://127.0.0.1:8100/api/health`; options: `GET /api/options`; prediction: `POST /api/predict`; Swagger: `http://127.0.0.1:8100/docs`. Run API tests from `backend/` after training with `python -m pip install pytest httpx` and `python -m pytest`.
