import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from scipy import stats
from evidently.core.report import Report
from evidently.presets import DataDriftPreset

FEATURES_PARQUET = Path("data/features/features.parquet")
PREDICTION_LOG = Path("data/logs/predictions.jsonl")
REPORTS_DIR = Path("reports")
DRIFT_THRESHOLD = 0.3  # flag if >30% of columns show drift (KS p < 0.05)

MONITOR_COLS = [
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h",
    "forecast_kwh_mean",
]


def _ks_drift_flag(reference: pd.DataFrame, current: pd.DataFrame, cols: list[str]) -> bool:
    drifted = sum(
        1 for col in cols
        if stats.ks_2samp(reference[col].dropna(), current[col].dropna()).pvalue < 0.05
    )
    return drifted / max(len(cols), 1) > DRIFT_THRESHOLD


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
) -> tuple[Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in MONITOR_COLS if c in reference.columns and c in current.columns]

    report = Report([DataDriftPreset()], include_tests=False)
    snapshot = report.run(current[cols], reference[cols])

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = output_dir / f"drift_{date_str}.html"
    snapshot.save_html(str(report_path))

    drift_detected = _ks_drift_flag(reference, current, cols)
    return report_path, drift_detected


def run_drift_check(days_back: int = 7) -> None:
    reference = pd.read_parquet(FEATURES_PARQUET)

    if not PREDICTION_LOG.exists():
        print("No prediction log found. Generate predictions first via /predict.")
        return

    logs = []
    with open(PREDICTION_LOG) as f:
        for line in f:
            logs.append(json.loads(line.strip()))

    if not logs:
        print("Prediction log is empty.")
        return

    current = pd.DataFrame(logs)
    current["logged_at"] = pd.to_datetime(current["logged_at"])
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    current = current[current["logged_at"] >= cutoff]

    if len(current) < 30:
        print(f"Only {len(current)} predictions in last {days_back} days. Need >=30 for reliable drift detection.")
        return

    report_path, drift_detected = compute_drift_report(reference, current)
    print(f"Drift report saved to {report_path}")
    if drift_detected:
        print("DRIFT DETECTED — consider retraining. Run: python -m src.training.train && python -m src.training.promote")
    else:
        print("No significant drift detected.")


if __name__ == "__main__":
    run_drift_check()
