import pandas as pd
import numpy as np
import pytest
from src.pipeline.features import build_features, FEATURE_COLS


def make_hourly_df(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2007-01-08 00:00", periods=n, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({"consumption_kwh": rng.uniform(0.2, 5.0, n)}, index=idx)


def test_build_features_returns_all_feature_columns():
    df = make_hourly_df(300)
    result = build_features(df)
    for col in FEATURE_COLS + ["target"]:
        assert col in result.columns, f"Missing column: {col}"


def test_build_features_drops_nan_rows():
    df = make_hourly_df(300)
    result = build_features(df)
    assert result.isna().sum().sum() == 0


def test_build_features_lag_values_are_correct():
    idx = pd.date_range("2007-01-08 00:00", periods=300, freq="h")
    values = list(range(300))
    df = pd.DataFrame({"consumption_kwh": values}, index=idx)
    result = build_features(df)
    # lag_24h != lag_1h because they reference different offsets
    assert (result["lag_1h"] != result["lag_24h"]).any()


def test_build_features_calendar_hour_range():
    df = make_hourly_df(300)
    result = build_features(df)
    assert result["hour"].between(0, 23).all()
    assert result["day_of_week"].between(0, 6).all()
    assert result["is_weekend"].isin([0, 1]).all()
    assert result["is_holiday"].isin([0, 1]).all()


def test_build_features_minimum_rows_required():
    # Need at least 170 rows; fewer should raise ValueError
    df = make_hourly_df(10)
    with pytest.raises(ValueError, match="at least 170"):
        build_features(df)
