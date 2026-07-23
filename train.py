"""Train a custom Ultralytics YOLO model from a dataset YAML file."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Ultralytics data.yaml")
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--project", default="runs/waste")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    model = YOLO(args.base_model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        project=args.project,
        name=args.name,
        device=args.device,
        seed=42,
        deterministic=True,
        patience=10,
    )


if __name__ == "__main__":
    main()
