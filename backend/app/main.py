from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import FRONTEND_ORIGINS
from .predictor import get_predictor, options, predict
from .schemas import PredictionRequest, PredictionResponse

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fail startup instead of advertising a healthy API without a usable model.
    get_predictor()
    yield


app = FastAPI(title="Real Estate Price Prediction API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=FRONTEND_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/api/health")
def health():
    try:
        get_predictor()
    except Exception:
        return {"status": "error", "model_loaded": False}
    return {"status": "ok", "model_loaded": True}


@app.get("/api/options")
def get_options():
    try:
        return options()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Prediction model is unavailable") from exc


@app.post("/api/predict", response_model=PredictionResponse)
def api_predict(payload: PredictionRequest):
    try:
        return predict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Prediction model is unavailable") from exc
