"""Basic unsupervised clustering with K-Means."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def run(output_dir: Path | str = "artifacts") -> dict[str, object]:
    dataset = load_wine()

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "cluster",
                KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE),
            ),
        ]
    )
    cluster_ids = model.fit_predict(dataset.data)
    scaled_features = model.named_steps["scale"].transform(dataset.data)

    metrics: dict[str, object] = {
        "task": "clustering",
        "dataset": "sklearn_wine",
        "rows": len(dataset.data),
        "clusters": 3,
        "silhouette_score": silhouette_score(scaled_features, cluster_ids),
        "adjusted_rand_index": adjusted_rand_score(dataset.target, cluster_ids),
        "cluster_counts": {
            str(cluster_id): int((cluster_ids == cluster_id).sum())
            for cluster_id in sorted(set(cluster_ids))
        },
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "clustering_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
