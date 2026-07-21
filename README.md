# Smart Waste Detector with YOLO and Streamlit

A computer-vision demo that runs a custom Ultralytics YOLO model on uploaded images or a local webcam feed, displays detected waste classes, and maps each class to a configurable dustbin category.

> **Status:** educational prototype. Detection quality depends entirely on the training data and the included `best.pt` checkpoint. Dustbin colours and disposal rules vary by city and country; verify local guidance before using the recommendations.

## Features

- Image upload and annotated detections
- Local webcam inference
- Adjustable confidence and IoU thresholds
- Per-class filtering
- Rule-based waste-class-to-bin mapping
- Session detection log
- Waste-count dashboard
- Downloadable annotated image and CSV session log

## Tech stack

- Python
- Streamlit
- Ultralytics YOLO
- OpenCV
- Pillow
- NumPy
- Pandas

## Repository requirements

The application expects a compatible custom model checkpoint named:

```text
best.pt
```

The checkpoint must use class names that match the waste categories expected by the interface. This repository does not automatically train a model when the application starts.

## Setup

```bash
git clone https://github.com/PseudoOzone/Waste-Detection-using-YOLOv11.git
cd Waste-Detection-using-YOLOv11

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print the local application URL in the terminal.

## Usage

### Image inference

1. Open the **Image Upload** tab.
2. Upload a JPG, JPEG, or PNG image.
3. Adjust confidence, IoU, or selected classes in the sidebar.
4. Review the annotated image and class-to-bin suggestions.
5. Download the image or session log if needed.

### Webcam inference

1. Run the application on a machine with a locally accessible webcam.
2. Open the **Webcam** tab.
3. Enable the webcam checkbox.
4. Disable it when finished so the camera can be released.

Browser-hosted or cloud deployments may not be able to access `cv2.VideoCapture(0)` on the user's device. The current webcam implementation is intended primarily for local execution.

## Bin mapping

The default mapping is stored in `DUSTBIN_MAP` inside `app.py`.

```python
DUSTBIN_MAP = {
    "food": "Brown",
    "organic": "Brown",
    "plastic": "Blue",
    "glass": "Green",
    "metal": "Blue",
    "paper": "Blue",
    "hazardous": "Red",
    "e-waste": "Red",
    "non-recyclable": "Black",
}
```

Update this mapping for the municipality or organization where the demo is being used.

## Known limitations

- The application fails at startup if `best.pt` is absent or incompatible.
- Uploaded files are decoded in memory and are not protected by explicit pixel-count or decompression-bomb limits.
- Webcam inference runs in a Streamlit loop and may block responsive reruns on some systems.
- A detection can be appended to the session log on every video frame, causing rapid memory growth.
- Class-to-bin assignment uses substring matching and can misclassify unfamiliar labels.
- No benchmark, confusion matrix, mAP result, dataset card, or model card is currently included.
- The application does not identify whether an item is clean, contaminated, recyclable in a specific region, or safe to handle.

## Recommended evaluation additions

Before presenting model performance, add:

- dataset source and license
- class counts and train/validation/test split
- annotation policy
- per-class precision and recall
- mAP50 and mAP50-95
- confusion matrix
- examples of false positives and false negatives
- model checkpoint provenance and training configuration

## Safety

Do not use the detector to make hazardous-material handling decisions. Batteries, chemicals, medical waste, sharp objects, and electronic waste require local safety procedures even when the model assigns a bin colour.

## License

Educational and portfolio use only unless a separate license file states otherwise.
