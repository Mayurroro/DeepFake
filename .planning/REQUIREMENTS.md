# Requirements

## Validated
- ✓ Execute basic deepfake detection feature extraction (FFT, Canny, Mel-spectrograms)
- ✓ Streamlit frontend + FastAPI backend connectivity
- ✓ Unified ensemble model structure (`DeepfakeEnsemble`)

## Active
- [ ] Ensure dataset paths are correctly configured for training.
- [ ] Update or verify CUDA availability in `train.py` parameters.
- [ ] Retrain `ImageDetector` utilizing GPU with proper batch size and learning rate optimizations for better accuracy.
- [ ] Retrain `AudioClassifier` utilizing GPU.
- [ ] Verify that model sizes/shapes match expected dimensions in inference scripts.
- [ ] Update `weights/` directory with the new `deepfake_model_latest.pth`.

## Out of Scope
- Architectural overhaul of the models (sticking to the current `EfficientNet-B4` and `CNN-LSTM` base unless trivially optimizable).
- Retraining the Video model (Video inference is a known stub and requires separate feature work, per CONCERNS.md).
