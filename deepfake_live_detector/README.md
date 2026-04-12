# 🛡️ Deepfake Live Detector

Real-time AI-powered deepfake detection for **images** and **audio** with live webcam + microphone capture.

---

## 📁 Project Structure

```
deepfake_live_detector/
├── models/
│   ├── image_detector.py      # EfficientNet-B4 + FFT/Edge heads
│   └── audio_detector.py      # CNN-LSTM on Mel-spectrograms
├── utils/
│   ├── feature_extractors.py  # Image & audio preprocessing
│   └── visualization.py       # Grad-CAM heatmaps, confidence bars
├── realtime/
│   ├── camera_capture.py      # Thread-safe OpenCV webcam
│   ├── mic_stream.py          # Rolling-buffer microphone stream
│   └── live_inference.py      # GPU/CPU inference + reasoning engine
├── api/
│   ├── fastapi_server.py      # /detect and /health REST endpoints
│   └── batch_processor.py     # Concurrent batch file scanner
├── frontend/
│   └── streamlit_app.py       # Dashboard: Camera · Mic · Upload · History
├── training/
│   ├── prepare_datasets.py    # Dataset crawler for d:\VIT\Datasets
│   └── train.py               # Unified train + evaluate script
├── export_onnx.py             # One-click ONNX export
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd d:\VIT\deepfake_live_detector
pip install -r requirements.txt
```

### 2. Train & Save Models
```bash
# Train BOTH image + audio models (auto-prepares datasets first)
python training/train.py --mode all --epochs 30 --batch 32

# Train image model only
python training/train.py --mode image --epochs 30 --batch 32

# Train audio model only
python training/train.py --mode audio --epochs 30 --batch 32

# Evaluate only (no training)
python training/train.py --mode all --eval-only

# Skip dataset preparation (reuse existing lists)
python training/train.py --mode all --epochs 30 --skip-prepare
```

**Models are saved to:**
```
weights/
├── image_detector.pth        # Best image model checkpoint
├── image_detector_final.pth  # Final epoch image model
├── audio_detector.pth        # Best audio model checkpoint
└── audio_detector_final.pth  # Final epoch audio model
```

### 3. Launch Dashboard
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```
Open http://localhost:8501 → Use Camera, Microphone, or Upload tabs.

### 4. Launch API Server
```bash
uvicorn api.fastapi_server:app --host 0.0.0.0 --port 8000
```
API docs at http://localhost:8000/docs

### 5. Batch Processing
```bash
# Start API server first, then:
python api/batch_processor.py --dir path/to/files --out results.json --workers 10
```

### 6. Export to ONNX
```bash
python export_onnx.py
# Creates: onnx_models/image_detector.onnx, onnx_models/audio_detector.onnx
```

---

## 🗂️ Datasets

The training script auto-crawls `d:\VIT\Datasets`:
- `DeepFake images/` → `fake/` and `real/` subfolders (train + test splits)
- `DeepFake audio/` → `fake/` and `real/` WAV files (80/20 auto-split)

---

## 🧠 Detection Reasoning

When content is flagged, the system explains **why** with measured feature analysis:

**Image checks:** Frequency (FFT), Edge sharpness, Texture uniformity, Lighting, Color distribution, EXIF metadata

**Audio checks:** Spectral centroid, Zero-crossings, MFCC formants, Spectral rolloff, Dynamic range, Breath/pause detection

---

## 🐳 Docker
```bash
docker build -t deepfake-detector .
docker run -p 8501:8501 -p 8000:8000 deepfake-detector
```
