# STACK.md — Technology Stack

## Language & Runtime

- **Python 3.x** — primary language for all modules
- **No JavaScript/TypeScript** — pure Python stack; no Node.js
- Runtime: standard CPython; GPU-accelerated via CUDA (NVIDIA) or MPS (Apple Silicon) or CPU fallback

## Deep Learning Framework

| Library | Version | Purpose |
|---|---|---|
| `torch` | ≥2.0.0 | Core ML framework (PyTorch) |
| `torchvision` | ≥0.15.0 | Image transforms, pretrained backbones |
| `torchaudio` | ≥2.0.0 | Audio utilities |
| `timm` | ≥0.9.2 | Pretrained model zoo (EfficientNet-B4 backbone) |
| `einops` | ≥0.6.1 | Tensor reshaping (deepfake_detector only) |

## Computer Vision

| Library | Version | Purpose |
|---|---|---|
| `opencv-python` / `opencv-python-headless` | ≥4.8.0 | Frame extraction, FFT, edge detection, optical flow |
| `Pillow` | ≥10.0.0 | Image loading, EXIF metadata reading |
| `albumentations` | ≥1.3.1 | Data augmentation during training |

## Audio Processing

| Library | Version | Purpose |
|---|---|---|
| `librosa` | ≥0.10.0 | Mel-spectrogram, MFCC, spectral features |
| `soundfile` | ≥0.12.1 | Audio file I/O |
| `PyAudio` | ≥0.2.14 | Real-time microphone capture (live_detector only) |
| `sounddevice` | ≥0.4.6 | Audio device streaming (live_detector only) |

## Web Framework & API

| Library | Version | Purpose |
|---|---|---|
| `fastapi` | ≥0.103.1 | REST API server (`/detect` endpoint) |
| `uvicorn` | ≥0.23.2 | ASGI server for FastAPI |
| `pydantic` | ≥2.3.0 | Request/response data validation |
| `python-multipart` | ≥0.0.6 | File upload handling |

## Frontend / Dashboard

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.26.0 | Web dashboard UI (both modules) |
| `streamlit-webrtc` | ≥0.47.0 | WebRTC webcam/mic integration (live_detector only) |

## Training & Data Science Utilities

| Library | Version | Purpose |
|---|---|---|
| `scikit-learn` | ≥1.3.0 | Metrics, evaluation utilities |
| `numpy` | ≥1.24.0 | Numerical arrays |
| `pandas` | ≥2.0.0 | Tabular data in dashboard history |
| `scipy` | ≥1.10.0 | Scientific computations |
| `matplotlib` | ≥3.7.2 | Plotting (training diagnostics) |
| `tqdm` | ≥4.66.1 | Training progress bars |
| `PyYAML` | ≥6.0.1 | Config file parsing (deepfake_detector) |
| `jupyter` | ≥1.0.0 | Notebook support (deepfake_detector only) |

## ONNX / Export (live_detector only)

| Library | Version | Purpose |
|---|---|---|
| `onnx` | ≥1.14.0 | Model export to ONNX format |
| `onnxruntime` | ≥1.15.0 | Optimized inference with ONNX |

## Containerization

- **Docker** — both modules have Dockerfiles
- Base image: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- System deps: `ffmpeg`, `libsndfile1`, `libgl1-mesa-glx`, `portaudio19-dev` (live_detector)
- Exposed ports: `8000` (FastAPI), `8501` (Streamlit)

## Configuration & Key Files

- `requirements.txt` — Python dependencies (per module, not monorepo)
- `Dockerfile` — Container build (per module)
- No `.env` files, `pyproject.toml`, or `setup.py` found
- `deepfake_training.ipynb` — Jupyter notebook in `deepfake_detector/`
- `deepfake_live_detector/export_onnx.py` — ONNX export script
- `deepfake_live_detector/debug_train.py` — Training debug helper

## GPU Acceleration

Both modules auto-detect and use in priority order:
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon) — live_detector only
3. CPU (fallback)

Example from `realtime/live_inference.py`:
```python
if torch.cuda.is_available():
    _device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _device = torch.device("mps")
else:
    _device = torch.device("cpu")
```
