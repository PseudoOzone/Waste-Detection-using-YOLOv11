# Waste Detector Model Card

No model checkpoint is committed by default.

A checkpoint should be added locally as `models/best.pt` only after its provenance is documented. Do not claim ownership of a checkpoint downloaded from another repository or training platform.

## Required information before publishing results

- model architecture and base checkpoint
- dataset name, source, license, and version
- annotation policy and class definitions
- number of images and objects per class
- train, validation, and test split method
- augmentation settings
- image size, batch size, epochs, optimiser, and seed
- mAP50 and mAP50-95
- per-class precision and recall
- confusion matrix
- examples of false positives and false negatives
- intended uses and prohibited uses

## Intended use

Educational object-detection experiments and local portfolio demonstrations.

## Not intended for

- hazardous-material handling decisions
- municipal compliance decisions
- autonomous sorting machinery without independent validation
- claims that an item is safe, clean, recyclable, or non-toxic

## Attribution

The application code in this repository is maintained separately from any future dataset or model checkpoint. Every dataset and checkpoint must retain its own license, provenance, and attribution.
