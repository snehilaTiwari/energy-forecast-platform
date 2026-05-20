"""
End-to-end smoke test on synthetic data. Requires no running services.
Verifies: feature engineering → train/eval loop produces valid metrics.
"""
import pytest
import pandas as pd
import numpy as np
from src.pipeline.features import build_features
from src.training.train import chronological_split, compute_metrics, FEATURE_COLS, TARGET

pytestmark = pytest.mark.slow

def test_full_pipeline_on_sample():
    rng = np.random.default_rng(99)
    idx = pd.date_range("2010-01-08", periods=1000, freq="h")
    df_clean = pd.DataFrame({"consumption_kwh": rng.uniform(0.3, 4.5, 1000)}, index=idx)

    df_features = build_features(df_clean)
    assert len(df_features) > 0
    for col in FEATURE_COLS + ["target"]:
        assert col in df_features.columns

    train_df, val_df, test_df = chronological_split(df_features)
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(n_estimators=10, verbosity=-1)
    model.fit(train_df[FEATURE_COLS], train_df[TARGET])
    preds = model.predict(test_df[FEATURE_COLS])
    metrics = compute_metrics(test_df[TARGET].values, preds)

    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert "mape" in metrics
