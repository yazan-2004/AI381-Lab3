
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "model.joblib"

# MLflow tracking database
mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("AI381-Lab3-Iris")


def train():
    # Load dataset
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    # Define models
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                random_state=42,
            ),
        ),
    }

    results = []

    # Train and evaluate each model
    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(
                y_test,
                predictions,
                average="macro",
            )

            recall = recall_score(
                y_test,
                predictions,
                average="macro",
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="macro",
            )

            # Log parameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
            })

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
            )

            results.append({
                "name": name,
                "model": model,
                "accuracy": accuracy,
                "f1_macro": f1,
            })

            print(f"\nModel: {name}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

    # Select best model using F1, then accuracy
    champion = max(
        results,
        key=lambda result: (
            result["f1_macro"],
            result["accuracy"],
        ),
    )

    # Save champion model
    joblib.dump(champion["model"], MODEL_PATH)

    print("\n" + "=" * 40)
    print("CHAMPION MODEL")
    print("=" * 40)
    print(f"Name: {champion['name']}")
    print(f"Accuracy: {champion['accuracy']:.4f}")
    print(f"F1 Score: {champion['f1_macro']:.4f}")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
