"""Evaluate a trained Ultralytics YOLO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--data", required=True, help="Path to Ultralytics data.yaml")
    parser.add_argument("--output", default="artifacts/yolo_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    metrics = YOLO(str(model_path)).val(data=str(data_path), verbose=False)
    summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
