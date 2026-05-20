import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

FEATURES_PARQUET = Path("data/features/features.parquet")
FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h",
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
]
TARGET = "target"
MODEL_NAME = "EnergyForecast"
MLFLOW_URI = "http://localhost:5000"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _train_and_log(model, run_name: str, X_train, y_train, X_val, y_val, X_test, y_test, params: dict) -> dict:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        metrics = compute_metrics(y_test.values, model.predict(X_test))
        mlflow.log_metrics(metrics)
        if "lightgbm" in run_name.lower():
            mlflow.lightgbm.log_model(model, name="model", registered_model_name=MODEL_NAME)
        else:
            mlflow.xgboost.log_model(model, name="model", registered_model_name=MODEL_NAME)
        return metrics


def train():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("energy-forecast")

    df = pd.read_parquet(FEATURES_PARQUET)
    train_df, val_df, test_df = chronological_split(df)
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET]

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "verbosity": -1,
    }
    lgb_metrics = _train_and_log(
        lgb.LGBMRegressor(**lgb_params), "lightgbm",
        X_train, y_train, X_val, y_val, X_test, y_test, lgb_params,
    )

    xgb_params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
    }
    xgb_metrics = _train_and_log(
        xgb.XGBRegressor(**xgb_params), "xgboost",
        X_train, y_train, X_val, y_val, X_test, y_test, xgb_params,
    )

    print(f"LightGBM — RMSE: {lgb_metrics['rmse']:.4f}  MAE: {lgb_metrics['mae']:.4f}")
    print(f"XGBoost  — RMSE: {xgb_metrics['rmse']:.4f}  MAE: {xgb_metrics['mae']:.4f}")


if __name__ == "__main__":
    train()
