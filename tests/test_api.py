# tests/test_api.py
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from fastapi.testclient import TestClient

MOCK_MODEL = MagicMock()
MOCK_MODEL.predict.return_value = np.array([0.5, 0.6, 0.7])

@pytest.fixture
def client():
    with patch("src.serving.app._load_model", return_value=(MOCK_MODEL, "3")):
        from src.serving.app import app
        with TestClient(app) as c:
            yield c

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "3"

PREDICT_PAYLOAD = {
    "timestamp": "2024-01-15T08:00:00",
    "lag_1h": 0.5, "lag_24h": 0.5, "lag_168h": 0.5,
    "lag_2h": 0.5, "lag_3h": 0.5, "lag_48h": 0.5,
    "rolling_mean_24h": 0.5, "rolling_mean_7d": 0.5, "rolling_std_24h": 0.1,
    "rolling_max_24h": 0.8, "rolling_min_24h": 0.2, "rolling_range_24h": 0.6,
}

def test_predict_returns_forecasts(client):
    response = client.post("/predict", json={**PREDICT_PAYLOAD, "periods": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data["forecasts"]) == 3
    assert data["forecasts"][0]["forecast_kwh"] == pytest.approx(0.5)

def test_predict_rejects_periods_above_168(client):
    response = client.post("/predict", json={**PREDICT_PAYLOAD, "periods": 169})
    assert response.status_code == 422

def test_predict_logs_prediction(client, tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    response = client.post("/predict", json={**PREDICT_PAYLOAD, "periods": 1})
    assert response.status_code == 200
    log_file = tmp_path / "predictions.jsonl"
    assert log_file.exists()
