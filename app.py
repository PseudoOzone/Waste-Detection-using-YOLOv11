"""Streamlit interface for the original waste-detection implementation."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.waste_detector import WasteDetector, load_bin_map, open_uploaded_image


MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", "models/best.pt"))
BIN_MAP_PATH = Path("config/waste_bins.json")


st.set_page_config(page_title="Waste Vision Lab", page_icon="♻️", layout="wide")


@st.cache_resource
def load_detector(model_path: str) -> WasteDetector:
    return WasteDetector(model_path, load_bin_map(BIN_MAP_PATH))


def render_detection_table(records: list[dict[str, object]]) -> None:
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        st.info("No objects crossed the selected confidence threshold.")


st.title("♻️ Waste Vision Lab")
st.caption(
    "Original Streamlit inference application for a custom Ultralytics YOLO checkpoint."
)

st.warning(
    "Disposal categories are broad educational mappings. Local municipal rules can differ."
)

with st.sidebar:
    st.header("Inference settings")
    confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.40, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.95, 0.50, 0.05)
    st.code(f"Model: {MODEL_PATH}")

try:
    detector = load_detector(str(MODEL_PATH))
except (FileNotFoundError, RuntimeError) as error:
    st.error(str(error))
    st.info(
        "Place your trained checkpoint at models/best.pt or set the "
        "YOLO_MODEL_PATH environment variable."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("Upload an image to run object detection.")
    st.stop()

try:
    image = open_uploaded_image(uploaded_file)
except ValueError as error:
    st.error(str(error))
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Input")
    st.image(image, use_container_width=True)

with st.spinner("Running detection..."):
    detections, annotated_image = detector.predict(
        image,
        confidence=confidence,
        iou=iou,
    )

records = [
    {
        "label": detection.label,
        "confidence": round(detection.confidence, 4),
        "disposal_category": detection.disposal_category,
        "box_xyxy": [round(value, 1) for value in detection.box_xyxy],
    }
    for detection in detections
]

with right:
    st.subheader("Model output")
    st.image(annotated_image, use_container_width=True)

st.subheader("Detections")
render_detection_table(records)

buffer = BytesIO()
Image.fromarray(annotated_image).save(buffer, format="PNG")
st.download_button(
    "Download annotated image",
    data=buffer.getvalue(),
    file_name="waste_detection_result.png",
    mime="image/png",
)
