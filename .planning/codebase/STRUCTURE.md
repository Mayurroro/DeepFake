# STRUCTURE.md — Directory Layout & Organization

## Repository Root (`d:\VIT\`)

```
d:\VIT\
├── .agent/                     # GSD agent skills and workflows
├── .git/                       # Git repository
├── .gitignore                  # Git ignore rules (minimal)
├── Datasets/                   # Training data (external, not tracked in git)
├── deepfake_detector/          # Module 1: Batch detection system
└── deepfake_live_detector/     # Module 2: Real-time detection system
```

**Note:** No monorepo tooling (no `pyproject.toml`, `setup.cfg`, `lerna.json`). Each module is fully self-contained.

---

## Module 1: deepfake_detector

```
deepfake_detector/
├── .git/                           # Nested git repo (this module may have been a separate repo)
├── Dockerfile                      # Container: CUDA 11.8, exposes 8000+8501
├── requirements.txt                # Python dependencies
├── deepfake_training.ipynb         # Jupyter notebook for interactive training
│
├── app.py                          # 🚪 Entry point: Streamlit UI frontend
│                                   #    Calls http://localhost:8000/detect
│
├── inference/
│   ├── api_server.py               # 🚪 Entry point: FastAPI /detect endpoint
│   └── batch_processor.py          # Batch file processing utility
│
├── models/
│   ├── __init__.py                 # Exports: ImageDetector, VideoAnalyzer,
│   │                               #          AudioClassifier, DeepfakeEnsemble
│   ├── ensemble.py                 # DeepfakeEnsemble (main model, routes modalities)
│   ├── image_detector.py           # ImageDetector (EfficientNet-B4 + freq/edge heads)
│   ├── video_analyzer.py           # VideoAnalyzer (ImageDetector + TransformerEncoder)
│   └── audio_classifier.py         # AudioClassifier (CNN + LSTM)
│
├── training/
│   ├── dataset_loader.py           # Dataset loading for ensemble training
│   ├── prepare_datasets.py         # Crawls Datasets/ and writes list files
│   ├── run_training.py             # Training runner script
│   ├── train_ensemble.py           # Ensemble training loop
│   ├── image_train_list.txt        # 🗂 Image paths + labels (train split)
│   ├── image_test_list.txt         # 🗂 Image paths + labels (test split)
│   ├── audio_train_list.txt        # 🗂 Audio paths + labels (train split)
│   └── audio_test_list.txt         # 🗂 Audio paths + labels (test split)
│
└── utils/
    ├── feature_extractors.py       # preprocess_image, extract_frames,
    │                               # extract_audio_features, FFT, edge, metadata
    └── visualization.py            # Plotting utilities (training/eval visuals)
```

---

## Module 2: deepfake_live_detector

```
deepfake_live_detector/
├── Dockerfile                      # Container: CUDA 11.8 + portaudio (for mic)
├── requirements.txt                # Python dependencies (includes PyAudio, ONNX)
├── README.md                       # Project documentation
├── export_onnx.py                  # 🔧 ONNX model export script
├── debug_train.py                  # 🔧 Training debug helper
│
├── frontend/
│   └── streamlit_app.py            # 🚪 Entry point: Full-featured Streamlit dashboard
│                                   #    Tabs: Camera | Microphone | Upload | History
│                                   #    Calls predict_image/predict_audio directly
│
├── realtime/
│   ├── __init__.py
│   ├── live_inference.py           # 📦 Core inference module (singleton lazy loading)
│   │                               #    predict_image() / predict_audio()
│   │                               #    + feature-level analysis (6 checks per modality)
│   ├── camera_capture.py           # Camera device helper
│   └── mic_stream.py               # Microphone stream helper
│
├── models/
│   ├── __init__.py                 # Exports: ImageDetector, AudioDetector
│   ├── image_detector.py           # ImageDetector (mirrors deepfake_detector version)
│   └── audio_detector.py           # AudioDetector (separate from AudioClassifier)
│
├── training/
│   ├── __init__.py
│   ├── prepare_datasets.py         # Crawl Datasets/ for image/audio paths
│   ├── run_training.py             # Simple training launcher
│   ├── train.py                    # 🚪 Full training+eval script with argparse
│   │                               #    --mode [image|audio|all] --epochs --batch --lr
│   ├── image_train_list.txt        # 🗂 Image list (train split)
│   ├── image_test_list.txt         # 🗂 Image list (test split)
│   ├── audio_train_list.txt        # 🗂 Audio list (train split, ~1.4 MB)
│   └── audio_test_list.txt         # 🗂 Audio list (test split, ~363 KB)
│
├── utils/
│   ├── __init__.py
│   ├── feature_extractors.py       # Shared: preprocess_image, extract_audio_features
│   └── visualization.py            # (currently basic, minimal use)
│
├── weights/                        # Model weights output directory
│   └── (image_detector.pth,        # Saved by training; loaded by live_inference.py
│         audio_detector.pth, etc.)
│
└── tmp/                            # Temp files for uploaded/recorded media
    └── (mic_recording.wav, etc.)   # Cleaned up after each session
```

---

## Datasets Directory

```
Datasets/                           # External data; not tracked by git
└── (subdirectories by class)       # e.g., real/, fake/, ai_generated/, audio/
```
Dataset paths are hardcoded or crawled dynamically by `prepare_datasets.py`.

---

## Key Locations Quick Reference

| What | Path |
|---|---|
| Streamlit UI (detector) | `deepfake_detector/app.py` |
| REST API server | `deepfake_detector/inference/api_server.py` |
| Ensemble model | `deepfake_detector/models/ensemble.py` |
| Streamlit UI (live) | `deepfake_live_detector/frontend/streamlit_app.py` |
| Live inference engine | `deepfake_live_detector/realtime/live_inference.py` |
| Model weights (live) | `deepfake_live_detector/weights/` |
| Training script (full) | `deepfake_live_detector/training/train.py` |
| Feature extraction | `*/utils/feature_extractors.py` |

---

## Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase` (e.g., `ImageDetector`, `DeepfakeEnsemble`, `ListDataset`)
- **Functions**: `snake_case` (e.g., `predict_image`, `extract_audio_features`)
- **Private helpers**: prefix `_` underscore (e.g., `_get_device`, `_load_image_model`, `_analyze_image_features`)
- **Constants**: `UPPER_CASE` (e.g., `CLASSES`, `WEIGHTS`, `ROOT`)
- **Module-level singletons**: prefix `_` underscore (e.g., `_device`, `_img_model`, `_aud_model`)
