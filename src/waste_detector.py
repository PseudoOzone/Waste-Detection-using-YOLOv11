"""Reusable YOLO inference utilities for the waste-detection demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


DEFAULT_BIN_MAP = {
    "food": "Organic waste",
    "organic": "Organic waste",
    "paper": "Dry recyclable",
    "plastic": "Dry recyclable",
    "metal": "Dry recyclable",
    "glass": "Dry recyclable",
    "battery": "Hazardous or e-waste",
    "electronic": "Hazardous or e-waste",
    "e-waste": "Hazardous or e-waste",
    "hazardous": "Hazardous or e-waste",
}


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    disposal_category: str


def load_bin_map(path: Path | str | None = None) -> dict[str, str]:
    """Load a configurable label-to-disposal mapping."""
    if path is None:
        return dict(DEFAULT_BIN_MAP)

    mapping_path = Path(path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Bin mapping must be a non-empty JSON object.")

    return {str(key).lower(): str(value) for key, value in mapping.items()}


def disposal_category(label: str, mapping: Mapping[str, str]) -> str:
    """Map a model label to a broad disposal category."""
    normalized = label.strip().lower()
    for keyword, category in mapping.items():
        if keyword in normalized:
            return category
    return "Check local waste guidance"


def validate_image(image: Image.Image, max_pixels: int = 20_000_000) -> Image.Image:
    """Normalize an uploaded image and reject oversized inputs."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if width * height > max_pixels:
        raise ValueError(
            f"Image is too large ({width * height:,} pixels); limit is {max_pixels:,}."
        )
    return image.convert("RGB")


class WasteDetector:
    """Small wrapper around an Ultralytics YOLO checkpoint."""

    def __init__(
        self,
        model_path: Path | str,
        bin_mapping: Mapping[str, str] | None = None,
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. "
                "Train a model or set YOLO_MODEL_PATH to a valid checkpoint."
            )

        self.model = YOLO(str(path))
        self.bin_mapping = dict(bin_mapping or DEFAULT_BIN_MAP)

    def predict(
        self,
        image: Image.Image,
        confidence: float = 0.40,
        iou: float = 0.50,
    ) -> tuple[list[Detection], np.ndarray]:
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be in the interval (0, 1].")
        if not 0 < iou <= 1:
            raise ValueError("iou must be in the interval (0, 1].")

        rgb_image = validate_image(image)
        results = self.model.predict(
            source=np.asarray(rgb_image),
            conf=confidence,
            iou=iou,
            verbose=False,
        )
        result = results[0]
        detections: list[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            label = str(self.model.names[class_id])
            coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf.item()),
                    box_xyxy=coordinates,
                    disposal_category=disposal_category(label, self.bin_mapping),
                )
            )

        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        return detections, annotated_rgb


def open_uploaded_image(file_object: Any) -> Image.Image:
    """Open an uploaded file and return a detached PIL image."""
    try:
        image = Image.open(file_object)
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a valid image.") from error
    return validate_image(image)
