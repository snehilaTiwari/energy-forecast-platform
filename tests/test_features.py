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


def test_new_lag_features_differ_from_each_other():
    idx = pd.date_range("2007-01-08 00:00", periods=300, freq="h")
    values = list(range(300))
    df = pd.DataFrame({"consumption_kwh": values}, index=idx)
    result = build_features(df)
    assert (result["lag_2h"] != result["lag_1h"]).any()
    assert (result["lag_3h"] != result["lag_2h"]).any()
    assert (result["lag_48h"] != result["lag_24h"]).any()


def test_rolling_range_equals_max_minus_min():
    df = make_hourly_df(300)
    result = build_features(df)
    expected = (result["rolling_max_24h"] - result["rolling_min_24h"]).round(10)
    actual = result["rolling_range_24h"].round(10)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_hour_x_weekend_is_zero_on_weekday():
    # 2007-01-08 is a Monday — all rows are weekdays
    idx = pd.date_range("2007-01-08 00:00", periods=300, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"consumption_kwh": rng.uniform(0.2, 5.0, 300)}, index=idx)
    result = build_features(df)
    weekday_rows = result[result.index.dayofweek < 5]
    assert (weekday_rows["hour_x_weekend"] == 0).all()


def test_hour_x_weekend_matches_hour_on_weekend():
    # 2007-01-06 is a Saturday — first 48h are weekend
    idx = pd.date_range("2007-01-06 00:00", periods=300, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"consumption_kwh": rng.uniform(0.2, 5.0, 300)}, index=idx)
    result = build_features(df)
    weekend_rows = result[result.index.dayofweek >= 5]
    assert (weekend_rows["hour_x_weekend"] == weekend_rows.index.hour).all()
