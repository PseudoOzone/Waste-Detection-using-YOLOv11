"""Synthetic user-access anomaly detection with Isolation Forest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def make_access_data(
    normal_rows: int = 1_000,
    anomaly_rows: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)

    normal = np.column_stack(
        [
            rng.normal(40, 12, normal_rows).clip(1, None),
            rng.lognormal(3.5, 0.45, normal_rows),
            rng.normal(11, 3, normal_rows).clip(0, 23),
            rng.poisson(2, normal_rows),
            rng.normal(1.5, 0.6, normal_rows).clip(0, None),
        ]
    )
    anomalies = np.column_stack(
        [
            rng.normal(180, 35, anomaly_rows).clip(1, None),
            rng.lognormal(6.2, 0.55, anomaly_rows),
            rng.choice([0, 1, 2, 3, 22, 23], anomaly_rows),
            rng.poisson(16, anomaly_rows),
            rng.normal(5, 1.2, anomaly_rows).clip(1, None),
        ]
    )

    X = np.vstack([normal, anomalies])
    y = np.concatenate(
        [
            np.zeros(normal_rows, dtype=int),
            np.ones(anomaly_rows, dtype=int),
        ]
    )
    return X, y


def run(output_dir: Path | str = "artifacts") -> dict[str, object]:
    X, y = make_access_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    normal_training_rows = X_train[y_train == 0]
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "detector",
                IsolationForest(
                    n_estimators=300,
                    contamination=0.05,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(normal_training_rows)

    raw_predictions = model.predict(X_test)
    predictions = (raw_predictions == -1).astype(int)

    metrics: dict[str, object] = {
        "task": "anomaly_detection",
        "dataset": "synthetic_user_access_logs",
        "normal_training_rows": len(normal_training_rows),
        "test_rows": len(X_test),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "anomaly_detection_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
