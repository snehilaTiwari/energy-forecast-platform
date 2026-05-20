import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, Field

MODEL_NAME = "EnergyForecast"
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
PREDICTION_LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "data/logs/predictions.jsonl"))

FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h",
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
]

_model = None
_model_version: str = "unknown"


def _load_model():
    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    client = MlflowClient(MLFLOW_URI)
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    version = versions[0].version if versions else "unknown"
    return model, version


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version
    _model, _model_version = _load_model()
    yield


app = FastAPI(title="Energy Forecast API", lifespan=lifespan)


class PredictRequest(BaseModel):
    timestamp: datetime
    lag_1h: float
    lag_24h: float
    lag_168h: float
    rolling_mean_24h: float
    rolling_mean_7d: float
    rolling_std_24h: float
    periods: int = Field(default=1, ge=1, le=168)


class ForecastPoint(BaseModel):
    timestamp: datetime
    forecast_kwh: float


class PredictResponse(BaseModel):
    forecasts: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    last_retrain: str


def _build_feature_rows(req: PredictRequest) -> pd.DataFrame:
    rows = []
    for i in range(req.periods):
        ts = req.timestamp + pd.Timedelta(hours=i)
        rows.append({
            "lag_1h": req.lag_1h,
            "lag_24h": req.lag_24h,
            "lag_168h": req.lag_168h,
            "rolling_mean_24h": req.rolling_mean_24h,
            "rolling_mean_7d": req.rolling_mean_7d,
            "rolling_std_24h": req.rolling_std_24h,
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "month": ts.month,
            "is_weekend": int(ts.weekday() >= 5),
            "is_holiday": 0,
        })
    return pd.DataFrame(rows, columns=FEATURE_COLS)


def _log_prediction(req: PredictRequest, forecasts: list[ForecastPoint]) -> None:
    log_path = Path(os.getenv("PREDICTION_LOG_PATH", "data/logs/predictions.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "logged_at": datetime.utcnow().isoformat(),
        "request_timestamp": req.timestamp.isoformat(),
        "lag_1h": req.lag_1h,
        "lag_24h": req.lag_24h,
        "lag_168h": req.lag_168h,
        "rolling_mean_24h": req.rolling_mean_24h,
        "rolling_mean_7d": req.rolling_mean_7d,
        "rolling_std_24h": req.rolling_std_24h,
        "periods": req.periods,
        "forecast_kwh_mean": float(np.mean([f.forecast_kwh for f in forecasts])),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    X = _build_feature_rows(req)
    preds = _model.predict(X)
    forecasts = [
        ForecastPoint(
            timestamp=req.timestamp + pd.Timedelta(hours=i),
            forecast_kwh=float(preds[i]),
        )
        for i in range(req.periods)
    ]
    _log_prediction(req, forecasts)
    return PredictResponse(forecasts=forecasts)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=_model_version,
        last_retrain=datetime.utcnow().isoformat(),
    )
