# Waste Vision and ML Fundamentals Lab

An original learning-focused machine-learning portfolio maintained by Anshuman Bakshi. The repository now contains two parts:

1. A clean Streamlit application and reusable Python module for running a custom Ultralytics YOLO waste detector.
2. Four small scikit-learn projects covering classification, regression, clustering, and anomaly detection.

> **Repository history note:** GitHub may still display this repository as a fork because fork status is repository metadata. The application code, training/evaluation scripts, tests, ML-basics modules, and documentation on the current branch have been rewritten. A separate non-fork repository is required to remove the fork badge completely.

## Why this repository exists

The goal is not to present a downloaded checkpoint as original work. The goal is to demonstrate a complete, defensible ML workflow:

- define the task
- validate inputs
- train or load a model
- evaluate with appropriate metrics
- expose inference through an application
- document limitations and provenance
- write tests and automated checks

No trained waste checkpoint is committed by default because the previous checkpoint did not have sufficient dataset and training provenance. Add a checkpoint locally only after documenting how it was produced.

## Repository structure

```text
.
├── app.py                         # Streamlit image-inference interface
├── train.py                       # Reproducible Ultralytics training entry point
├── evaluate.py                    # mAP, precision, and recall export
├── src/
│   └── waste_detector.py          # Reusable model and image-validation logic
├── config/
│   └── waste_bins.json            # Editable disposal-category mapping
├── models/
│   └── .gitkeep                   # Put a local best.pt here
├── tests/
│   └── test_waste_detector.py
├── ml_basics/
│   ├── 01_classification.py
│   ├── 02_regression.py
│   ├── 03_clustering.py
│   ├── 04_anomaly_detection.py
│   └── README.md                  # Concepts and interview explanations
├── MODEL_CARD.md
└── .github/workflows/tests.yml
```

# Part 1: Waste-object detection

## What the application does

The Streamlit app:

- accepts JPG and PNG uploads
- validates image dimensions
- loads a configurable local YOLO checkpoint
- exposes confidence and IoU thresholds
- returns class labels, confidence values, bounding boxes, and broad disposal categories
- exports the annotated image

The disposal mapping is educational. Waste rules vary across cities and organisations.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/PseudoOzone/Waste-Detection-using-YOLOv11.git
cd Waste-Detection-using-YOLOv11

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run inference

Place a documented checkpoint at:

```text
models/best.pt
```

Then run:

```bash
streamlit run app.py
```

A different checkpoint path can be provided through:

```bash
YOLO_MODEL_PATH=/path/to/model.pt streamlit run app.py
```

## Train your own model

Prepare an Ultralytics-compatible `data.yaml`, then run:

```bash
python train.py \
  --data path/to/data.yaml \
  --base-model yolo11n.pt \
  --epochs 50 \
  --image-size 640 \
  --batch-size 16
```

The training script fixes the random seed and enables deterministic training where supported. Hardware and library versions can still affect results.

## Evaluate a checkpoint

```bash
python evaluate.py \
  --model models/best.pt \
  --data path/to/data.yaml
```

The evaluation script writes:

- mAP50
- mAP50-95
- mean precision
- mean recall

to `artifacts/yolo_metrics.json`.

## Object-detection concepts to understand

### Bounding box

A rectangle represented by four coordinates around a detected object.

### Confidence threshold

The minimum model confidence required to keep a prediction. A higher threshold usually reduces false positives but can miss real objects.

### IoU

Intersection over Union measures overlap between two boxes. It is used during evaluation and non-maximum suppression.

### Non-maximum suppression

When several boxes predict the same object, NMS keeps the strongest box and removes highly overlapping duplicates.

### Precision and recall

- Precision asks: of the objects predicted, how many were correct?
- Recall asks: of the real objects, how many were found?

### mAP

Mean Average Precision summarises precision-recall performance across classes. `mAP50-95` is stricter than `mAP50` because it averages across several IoU thresholds.

# Part 2: ML fundamentals

The `ml_basics` folder contains four executable examples built from scikit-learn datasets or synthetic data.

## 1. Classification

```bash
python ml_basics/01_classification.py
```

Uses logistic regression to predict a binary class. It teaches stratified splitting, scaling, class weights, precision, recall, F1, ROC-AUC, and confusion matrices.

## 2. Regression

```bash
python ml_basics/02_regression.py
```

Uses ridge regression to predict a continuous value. It teaches MAE, RMSE, R², coefficient regularisation, and the difference between numeric prediction and class prediction.

## 3. Clustering

```bash
python ml_basics/03_clustering.py
```

Uses K-Means without labels during training. It teaches unsupervised learning, centroids, feature scaling, silhouette score, and the limitations of choosing `k` in advance.

## 4. Anomaly detection

```bash
python ml_basics/04_anomaly_detection.py
```

Uses Isolation Forest on synthetic employee-access features. It teaches rare-event detection, baseline behaviour, false alerts, contamination assumptions, and the precision-recall trade-off. This example is particularly relevant to behavioural security products.

Read `ml_basics/README.md` for interview-ready explanations.

# Tests and automation

Run locally:

```bash
pytest -q
```

GitHub Actions performs:

- Python syntax checks
- unit tests
- execution smoke tests for all four basic ML projects

# Limitations

- No waste dataset or trained checkpoint is included by default.
- Disposal recommendations are broad examples, not municipal guidance.
- A detector cannot determine whether an object is contaminated, safe to handle, or recyclable in a particular location.
- The basic ML projects are educational and intentionally small.
- Synthetic anomaly detection is easier than real insider-threat detection.
- Metrics should only be quoted together with the dataset, split, seed, and experiment configuration.

# Attribution and responsible use

This repository was reworked as a learning portfolio with AI-assisted implementation and review. Anshuman should understand and be able to explain every public claim before presenting it in an interview.

Datasets, pretrained base models, and future checkpoints retain their own licenses and attribution requirements.
