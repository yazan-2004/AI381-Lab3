
import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
client = MlflowClient()

MODEL_NAME = "ChurnChampion"


def get_metric(run, possible_names):
    for name in possible_names:
        value = run.data.metrics.get(name)
        if value is not None:
            return value
    raise ValueError(f"Missing metric. Expected one of: {possible_names}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger-run-id", required=True)
    parser.add_argument("--min-f1-gain", type=float, default=0.005)
    parser.add_argument("--latency-budget-ms", type=float, default=100)
    args = parser.parse_args()

    champion_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="champion",
    )

    champion_run = client.get_run(champion_version.run_id)
    challenger_run = client.get_run(args.challenger_run_id)

    champion_f1 = get_metric(champion_run, ["f1"])
    challenger_f1 = get_metric(challenger_run, ["f1"])

    champion_latency = get_metric(
        champion_run,
        ["p95_latency_ms", "latency_p95_ms"],
    )
    challenger_latency = get_metric(
        challenger_run,
        ["p95_latency_ms", "latency_p95_ms"],
    )

    print("========== PROMOTION GATE ==========")
    print(f"Champion F1:    {champion_f1:.4f}")
    print(f"Challenger F1:  {challenger_f1:.4f}")
    print(f"Champion p95:   {champion_latency:.2f} ms")
    print(f"Challenger p95: {challenger_latency:.2f} ms")

    f1_passed = challenger_f1 >= champion_f1 + args.min_f1_gain
    latency_passed = challenger_latency <= args.latency_budget_ms

    if not f1_passed or not latency_passed:
        print("\nBLOCKED")
        if not f1_passed:
            print("Reason: Challenger did not achieve the minimum F1 gain.")
        if not latency_passed:
            print("Reason: Challenger exceeded the latency budget.")
        raise SystemExit(1)

    print("\nGate passed. Registering challenger...")

    registered = mlflow.register_model(
        model_uri=f"runs:/{args.challenger_run_id}/model",
        name=MODEL_NAME,
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=registered.version,
    )

    print("PROMOTED")
    print(f"New champion version: {registered.version}")


if __name__ == "__main__":
    main()
