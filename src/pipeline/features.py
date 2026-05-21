import pandas as pd
import holidays
from pathlib import Path

CLEAN_PARQUET = Path("data/raw/clean.parquet")
FEATURES_PARQUET = Path("data/features/features.parquet")

FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_168h",
    "lag_2h", "lag_3h", "lag_48h",
    "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h",
    "rolling_max_24h", "rolling_min_24h", "rolling_range_24h",
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "hour_x_weekend",
]

_FR_HOLIDAYS = holidays.France()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 170:
        raise ValueError("DataFrame must have at least 170 rows to compute lag_168h and target.")

    out = df[["consumption_kwh"]].copy()

    out["lag_1h"] = out["consumption_kwh"].shift(1)
    out["lag_24h"] = out["consumption_kwh"].shift(24)
    out["lag_168h"] = out["consumption_kwh"].shift(168)
    out["lag_2h"] = out["consumption_kwh"].shift(2)
    out["lag_3h"] = out["consumption_kwh"].shift(3)
    out["lag_48h"] = out["consumption_kwh"].shift(48)

    shifted = out["consumption_kwh"].shift(1)
    out["rolling_mean_24h"] = shifted.rolling(24).mean()
    out["rolling_mean_7d"] = shifted.rolling(168).mean()
    out["rolling_std_24h"] = shifted.rolling(24).std()
    out["rolling_max_24h"] = shifted.rolling(24).max()
    out["rolling_min_24h"] = shifted.rolling(24).min()
    out["rolling_range_24h"] = out["rolling_max_24h"] - out["rolling_min_24h"]

    out["hour"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)
    out["is_holiday"] = out.index.map(lambda dt: int(dt.date() in _FR_HOLIDAYS))
    out["hour_x_weekend"] = out.index.hour * (out.index.dayofweek >= 5).astype(int)

    out["target"] = out["consumption_kwh"].shift(-1)

    return out.dropna()


def run_feature_pipeline() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_PARQUET)
    result = build_features(df)
    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(FEATURES_PARQUET)
    print(f"Saved {len(result)} feature rows to {FEATURES_PARQUET}")
    return result


if __name__ == "__main__":
    run_feature_pipeline()
