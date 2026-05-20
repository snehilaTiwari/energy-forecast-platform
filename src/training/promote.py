import mlflow
from mlflow import MlflowClient

MODEL_NAME = "EnergyForecast"
IMPROVEMENT_THRESHOLD = 0.02
MLFLOW_URI = "http://localhost:5000"


def get_production_rmse(client: MlflowClient, model_name: str) -> float | None:
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        return None
    return float(client.get_run(versions[0].run_id).data.metrics["rmse"])


def get_best_challenger_rmse_and_version(client: MlflowClient, model_name: str) -> tuple[str, float] | None:
    versions = client.get_latest_versions(model_name, stages=["None"])
    if not versions:
        return None
    runs = [(v, float(client.get_run(v.run_id).data.metrics["rmse"])) for v in versions]
    best_version, best_rmse = min(runs, key=lambda x: x[1])
    return best_version.version, best_rmse


def should_promote(prod_rmse: float | None, challenger_rmse: float) -> bool:
    if prod_rmse is None:
        return True
    return (prod_rmse - challenger_rmse) / prod_rmse > IMPROVEMENT_THRESHOLD


def promote():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    prod_rmse = get_production_rmse(client, MODEL_NAME)
    result = get_best_challenger_rmse_and_version(client, MODEL_NAME)

    if result is None:
        print("No challenger versions found. Run train.py first.")
        return

    challenger_version, challenger_rmse = result

    if prod_rmse is not None:
        improvement = (prod_rmse - challenger_rmse) / prod_rmse
        print(f"Production RMSE : {prod_rmse:.4f}")
        print(f"Challenger RMSE : {challenger_rmse:.4f}  (version {challenger_version})")
        print(f"Improvement     : {improvement:.1%}  (threshold: {IMPROVEMENT_THRESHOLD:.0%})")
    else:
        print(f"No production model found. Promoting version {challenger_version} (RMSE={challenger_rmse:.4f})")

    if should_promote(prod_rmse, challenger_rmse):
        client.transition_model_version_stage(MODEL_NAME, challenger_version, "Production")
        print(f"Promoted version {challenger_version} to Production.")
    else:
        print("Challenger did not meet improvement threshold. Keeping current production model.")


if __name__ == "__main__":
    promote()
