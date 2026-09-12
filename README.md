---
title: Deepfake Live Detector
emoji: 🛡️
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 8501
pinned: false
---

# 🛡️ Deepfake Live Detector

Real-time AI-powered deepfake detection for **images** and **audio** with live webcam + microphone capture.

---

## 📁 Project Structure

```
deepfake_live_detector/
├── models/
│   ├── image_detector.py      # EfficientNet-B4 + FFT/Edge heads
│   ├── audio_detector.py      # CNN-LSTM on Mel-spectrograms
│   └── fusion.py              # Gated mixture-of-experts evidence fusion
├── utils/
│   ├── feature_extractors.py  # Image & audio preprocessing
│   ├── pipeline.py            # 10-stage hybrid forensic pipeline
│   ├── stage1_file_inspection.py
│   ├── stage2_provenance.py
│   ├── stage3_metadata.py
│   ├── stage4_quality.py
│   ├── stage5_classical.py
│   └── stage10_report.py
├── realtime/
│   ├── camera_capture.py      # Thread-safe OpenCV webcam
│   ├── mic_stream.py          # Rolling-buffer microphone stream
│   └── live_inference.py      # GPU/CPU inference + reasoning engine
├── api/
│   ├── fastapi_server.py      # /detect and /health REST endpoints
│   └── batch_processor.py     # Concurrent batch file scanner
├── frontend/
│   ├── streamlit_app.py       # Dashboard: Camera · Mic · Upload · History
│   └── index.html             # Static frontend assets
├── training/
│   ├── prepare_datasets.py    # Dataset crawler for d:\VIT\Datasets
│   ├── train.py               # Unified train + evaluate script
│   └── run_training.py        # One-command training orchestrator
├── export_onnx.py             # One-click ONNX export
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train & Save Models

```bash
# Train both image + audio models (auto-prepares datasets first)
python training/train.py --mode all --epochs 30 --batch 32

# Image only
python training/train.py --mode image --epochs 30 --batch 32

# Audio only
python training/train.py --mode audio --epochs 30 --batch 32

# Evaluate only (no training)
python training/train.py --mode all --eval-only

# Freeze backbone, train fusion head only
python training/train.py --mode all --freeze-backbone --epochs 20

# Skip dataset preparation (reuse existing lists)
python training/train.py --mode all --skip-prepare
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

## 🌐 One-Click Free Deploy (Hugging Face Spaces)

The dashboard runs in the browser at a public URL — no install, no GPU, no cost.

1. Generate the small ONNX runtime models once (they're what the CPU image serves):

   ```bash
   python export_onnx.py      # creates onnx_models/*.onnx (~40 MB)
   git add onnx_models && git commit && git push
   ```

2. Create a Space at https://huggingface.co/new-space → **Docker** → paste your GitHub repo (or push it).
   The README metadata (`sdk: docker`, `app_port: 8501`) tells Spaces how to run it.

3. Inference picks ONNX Runtime automatically (fast, ~300 MB RAM). Set `USE_ONNX=0` to force torch.

**CPU-only inference:** the export uses `onnx_models/*.onnx`; torch is only a fallback. Free-tier machines have no GPU — images/audio still analyze in under a second on CPU.

## 🐳 Docker

```bash
docker build -t deepfake-detector .          # slim CPU image (~1 GB, no CUDA)
docker run -p 8501:8501 -p 8000:8000 deepfake-detector
```

The image runs the Streamlit dashboard on port 8501. Start the API server separately by overriding the entrypoint:

```bash
docker run -p 8000:8000 deepfake-detector \
    uvicorn api.fastapi_server:app --host 0.0.0.0 --port 8000
```

The CPU image installs `requirements-serve.txt` (torch CPU wheels). For training on GPU, keep using the local env with `requirements.txt`, then re-export ONNX with `python export_onnx.py`.
