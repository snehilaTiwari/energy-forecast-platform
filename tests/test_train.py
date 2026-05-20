# tests/test_train.py
import numpy as np
import pandas as pd
import pytest
from src.training.train import compute_metrics, chronological_split, FEATURE_COLS

def make_feature_df(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2007-01-08", periods=n, freq="h")
    rng = np.random.default_rng(0)
    data = {col: rng.uniform(0, 1, n) for col in FEATURE_COLS}
    data["target"] = rng.uniform(0.2, 5.0, n)
    return pd.DataFrame(data, index=idx)

def test_compute_metrics_returns_rmse_mae_mape():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    metrics = compute_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"rmse", "mae", "mape"}
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["mape"] > 0

def test_chronological_split_proportions():
    df = make_feature_df(1000)
    train, val, test = chronological_split(df)
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150
    # Must be chronological — no overlap
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()

def test_compute_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y, y)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
