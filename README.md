# Deepfake Detector Retraining Project

This project focuses on retraining and enhancing existing deepfake detection models (Image and Audio) by utilizing newly available GPU hardware for higher accuracy and deeper training cycles.

## 📁 Repository Structure

- `deepfake_live_detector/` - Contains the real-time AI-powered deepfake detection system for images and audio, using webcam and microphone capture. Check the `README.md` inside this directory for specific running instructions.
- `Datasets/` - Audio and image dataset directories for training and evaluation.
- `.planning/` - GSD project planning and tracking files, including milestones, state, and codebase maps.

## 🎯 Core Value

Higher accuracy detection of AI-generated content through deeper and longer training cycles, made feasible by GPU acceleration.

## 🚀 Key Sub-Projects

### Live Detector & Retraining Pipeline
Located in `deepfake_live_detector/`. Contains the main logic for both batch and live inference, as well as the training pipeline:
- **Image Detector**: EfficientNet-B4 + FFT/Edge heads
- **Audio Detector**: CNN-LSTM on Mel-spectrograms
- **Frontend**: Streamlit-based real-time dashboard
- **API**: FastAPI server for remote detection endpoints

**Training**:
Make sure CUDA is properly configured and activated to utilize GPU acceleration. The `train.py` script inside `deepfake_live_detector/training/` is already configured for handling full end-to-end training and checkpoint saving for both image and audio.

```bash
cd deepfake_live_detector
pip install -r requirements.txt
python training/train.py --mode all --epochs 30 --batch 32
```

## 🐳 Docker Support
A Docker image could be built for the detector to prevent environment mismatches. See inside `deepfake_live_detector` for its `Dockerfile`.
