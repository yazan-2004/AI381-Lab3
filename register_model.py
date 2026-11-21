from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

client = MlflowClient()

experiment = client.get_experiment_by_name("AI381-Lab3-Churn")
if experiment is None:
    raise RuntimeError("Churn experiment not found")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.f1 DESC"],
)

if not runs:
    raise RuntimeError("No completed training runs found")

best_run = runs[0]

model_name = "ChurnChampion"

model_uri = f"runs:/{best_run.info.run_id}/model"

print("Registering model...")
print("Run:", best_run.info.run_name)
print("F1:", best_run.data.metrics["f1"])

registered = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
)

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=registered.version,
)

print("================================")
print("MODEL REGISTERED SUCCESSFULLY")
print("Model:", model_name)
print("Version:", registered.version)
print("Alias: champion")
print("Run ID:", best_run.info.run_id)
print("================================")