# PROJECT.md

## What This Is
Deepfake Detector Retraining — A project to retrain the existing deepfake detection models (Image and Audio) utilizing newly available GPU hardware to achieve better accuracy.

## Core Value
Higher accuracy detection of AI-generated content through deeper/longer training cycles made feasible by GPU acceleration.

## Success Looks Like
- `train.py` completes successfully using CUDA.
- The new models yield higher evaluation accuracy than the existing weights.
- The application (both batch and live detectors) successfully load and run the new `.pth` model weights.

## Key Principles
- **No architectural regressions**: We are retraining the existing model architectures (`ImageDetector` and `AudioClassifier`), not completely rewriting the ensemble logic.
- **Hardware leverage**: Ensure `device=cuda` is properly utilized in training loops.
- **Measurability**: Training should output clear accuracy metrics to compare against previous baselines.

## Evolution
This document evolves at phase transitions and milestone boundaries.
