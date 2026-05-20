# tests/test_drift.py
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from src.monitoring.drift import compute_drift_report, DRIFT_THRESHOLD, MONITOR_COLS

def make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lag_1h": rng.uniform(0.2, 5.0, n),
        "lag_24h": rng.uniform(0.2, 5.0, n),
        "lag_168h": rng.uniform(0.2, 5.0, n),
        "rolling_mean_24h": rng.uniform(0.2, 5.0, n),
        "rolling_mean_7d": rng.uniform(0.2, 5.0, n),
        "rolling_std_24h": rng.uniform(0.0, 1.0, n),
        "forecast_kwh_mean": rng.uniform(0.2, 5.0, n),
    })

def test_compute_drift_report_returns_html_path_and_flag(tmp_path):
    reference = make_df(300, seed=0)
    current = make_df(100, seed=0)  # same distribution → no drift
    report_path, drift_detected = compute_drift_report(reference, current, output_dir=tmp_path)
    assert report_path.exists()
    assert report_path.suffix == ".html"
    assert isinstance(drift_detected, bool)

def test_compute_drift_report_detects_shifted_distribution(tmp_path):
    reference = make_df(300, seed=0)
    current = make_df(100, seed=0)
    current["lag_1h"] = current["lag_1h"] + 10.0
    current["lag_24h"] = current["lag_24h"] + 10.0
    current["lag_168h"] = current["lag_168h"] + 10.0
    _, drift_detected = compute_drift_report(reference, current, output_dir=tmp_path)
    assert drift_detected is True

def test_compute_drift_report_no_drift_on_identical_data(tmp_path):
    reference = make_df(300, seed=42)
    _, drift_detected = compute_drift_report(reference, reference.copy(), output_dir=tmp_path)
    assert drift_detected is False
