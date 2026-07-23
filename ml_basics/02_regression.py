"""Basic regression with a leakage-safe scikit-learn pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def run(output_dir: Path | str = "artifacts") -> dict[str, object]:
    dataset = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics: dict[str, object] = {
        "task": "regression",
        "dataset": "sklearn_diabetes",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "regression_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
