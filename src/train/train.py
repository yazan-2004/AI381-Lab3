
from pathlib import Path
import hashlib
import tempfile
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "archive" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "churn_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("AI381-Lab3-Churn")


def calculate_data_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_latency_p95(model, X):
    latencies = []

    for index in range(len(X)):
        sample = X.iloc[[index]]

        start = time.perf_counter()
        model.predict(sample)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed * 1000)

    return float(np.percentile(latencies, 95))


def calculate_model_size(model):
    temp_path = ROOT / "temp_model_size.joblib"
    joblib.dump(model, temp_path)
    size_mb = temp_path.stat().st_size / (1024 * 1024)
    temp_path.unlink()
    return size_mb


def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Churn dataset not found: {DATA_PATH}"
        )

    print("Loading Churn dataset...")

    data = pd.read_csv(DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce",
    )

    data = data.dropna().copy()

    data["Churn"] = (
        data["Churn"].str.strip().str.lower() == "yes"
    ).astype(int)

    data = data.drop(columns=["customerID"])

    data_hash = calculate_data_hash(DATA_PATH)

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Dataset rows: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Data SHA256: {data_hash}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        preprocessing = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    StandardScaler(),
                    numerical_columns,
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns,
                ),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("classifier", classifier),
            ]
        )

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
            recall = recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0,
            )

            # Evaluate F1 for each Contract group.
            slice_scores = {}

            for contract in X_test["Contract"].unique():
                mask = X_test["Contract"] == contract

                slice_f1 = f1_score(
                    y_test[mask],
                    predictions[mask],
                    zero_division=0,
                )

                slice_scores[str(contract)] = float(slice_f1)

            weakest_slice_f1 = min(slice_scores.values())

            latency_p95 = calculate_latency_p95(
                model,
                X_test,
            )

            model_size_mb = calculate_model_size(model)

            signature = infer_signature(
                X_train.head(5),
                model.predict(X_train.head(5)),
            )

            mlflow.log_params({
                "model_name": name,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
            })

            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "weakest_slice_f1": weakest_slice_f1,
                "latency_p95_ms": latency_p95,
                "model_size_mb": model_size_mb,
            })

            mlflow.set_tag("data_sha256", data_hash)

            mlflow.log_dict(
                slice_scores,
                "contract_slice_f1.json",
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_train.head(5),
                serialization_format="cloudpickle",
            )

            results.append({
                "name": name,
                "model": model,
                "accuracy": accuracy,
                "f1": f1,
                "weakest_slice_f1": weakest_slice_f1,
                "latency_p95_ms": latency_p95,
                "model_size_mb": model_size_mb,
            })

            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1: {f1:.4f}")
            print(f"Weakest Contract F1: {weakest_slice_f1:.4f}")
            print(f"p95 latency: {latency_p95:.4f} ms")
            print(f"Model size: {model_size_mb:.4f} MB")

    champion = max(
        results,
        key=lambda result: (
            result["f1"],
            result["weakest_slice_f1"],
        ),
    )

    joblib.dump(champion["model"], MODEL_PATH)

    print("\n" + "=" * 45)
    print("CHURN CHAMPION MODEL")
    print("=" * 45)
    print(f"Name: {champion['name']}")
    print(f"F1: {champion['f1']:.4f}")
    print(
        f"Weakest Slice F1: "
        f"{champion['weakest_slice_f1']:.4f}"
    )
    print(
        f"p95 Latency: "
        f"{champion['latency_p95_ms']:.4f} ms"
    )
    print(f"Model Size: {champion['model_size_mb']:.4f} MB")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
