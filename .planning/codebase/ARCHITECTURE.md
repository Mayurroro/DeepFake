# ARCHITECTURE.md — System Architecture

## Overall Pattern

**Dual-Module Monorepo** — two standalone Python applications in the same Git repository, each independently deployable:

1. **`deepfake_detector/`** — Batch/file detection system (decoupled Streamlit + FastAPI)
2. **`deepfake_live_detector/`** — Real-time/live detection system (tightly coupled Streamlit + inline inference)

Both modules are self-contained: each has its own `requirements.txt`, `Dockerfile`, and independent model weights.

---

## Module 1: deepfake_detector

### Architecture: Decoupled 2-Tier (Frontend ↔ API)

```
┌─────────────────────────────────────┐
│  app.py (Streamlit UI)              │
│  - File upload (image/video/audio)  │
│  - POST http://localhost:8000/detect│
└──────────────┬──────────────────────┘
               │ HTTP (requests)
               ▼
┌─────────────────────────────────────┐
│  inference/api_server.py (FastAPI)  │
│  - POST /detect endpoint            │
│  - Routes by file ext               │
│  - Saves to ../tmp/, then deletes   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  models/ — DeepfakeEnsemble (nn.Module)                             │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ DeepfakeEnsemble                                             │   │
│  │  ├── ImageDetector (EfficientNet-B4 + freq_head + edge_head) │   │
│  │  ├── VideoAnalyzer (ImageDetector + TransformerEncoder)      │   │
│  │  ├── AudioClassifier (CNN + LSTM + aux_dense)                │   │
│  │  ├── fusion (Linear: 2016→512→3)                             │   │
│  │  └── temperature (scalar, calibration)                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  utils/feature_extractors.py        │
│  - preprocess_image()               │
│  - extract_frames()                 │
│  - extract_audio_features()         │
│  - extract_frequency_features()     │
│  - extract_edge_features()          │
│  - extract_metadata() (EXIF)        │
└─────────────────────────────────────┘
```

### Data Flow (Image Detection)
1. User uploads file → Streamlit → POST `/detect`
2. API saves to `../tmp/{filename}`
3. `preprocess_image()` → (rgb tensor, FFT tensor, edge tensor)
4. `model(image_tuple=(rgb, freq, edge))` → logits → softmax → class + confidence
5. Return `DetectionResponse` JSON (prediction, confidence, anomalies, processing_time_ms)
6. Temp file deleted; Streamlit renders result

### Data Flow (Video Detection)
- Currently a **stub**: calls `extract_frames()` but then returns hardcoded `("AI_GENERATED", 0.965, [...])` with `time.sleep(1.5)` — real video inference not implemented

---

## Module 2: deepfake_live_detector

### Architecture: Tightly Coupled Single-Process (Streamlit)

```
┌─────────────────────────────────────────────────────────────────┐
│  frontend/streamlit_app.py                                      │
│  ┌────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐           │
│  │Camera  │ │Microphone │ │  Upload  │ │ History  │           │
│  │tab     │ │tab        │ │  tab     │ │ tab      │           │
│  └───┬────┘ └─────┬─────┘ └────┬─────┘ └──────────┘           │
│      │            │             │                               │
│      └─────────── ▼─────────────┘                              │
│                   │ direct function call (no HTTP)              │
│                   ▼                                             │
│     realtime/live_inference.py                                  │
│     ┌─────────────────────────────────────────────┐            │
│     │  predict_image(source) / predict_audio(src) │            │
│     │  - Lazy model loading (singleton pattern)   │            │
│     │  - Feature-level analysis (6 checks/modality)│           │
│     │  - Returns: prediction, confidence,          │            │
│     │    anomalies, reasons, detection_time_ms     │            │
│     └──────────────┬──────────────────────────────┘            │
└────────────────────│────────────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
  models/image_detector.py  models/audio_detector.py
  (ImageDetector)            (AudioDetector — separate class
                              from AudioClassifier in detector)
         │
         └── utils/feature_extractors.py (shared)
```

### Lazy Loading Pattern (Singleton)
`live_inference.py` uses module-level globals for model caching:
```python
_device = None
_img_model = None
_aud_model = None

def _load_image_model():
    global _img_model
    if _img_model is None:
        _img_model = ImageDetector(pretrained=True).to(_get_device()).eval()
        # load weights if exist
    return _img_model
```
Models are loaded once per process lifetime; subsequent calls reuse the cached model.

---

## Ensemble Model Architecture (deepfake_detector)

### DeepfakeEnsemble — Modality Routing

```
Input → Route by modality:
  image_tuple → ImageDetector → (logits, features)
  video_tuple only → VideoAnalyzer → (logits, features)
  audio_tuple only → AudioClassifier → (logits, features)
  video + audio → VideoAnalyzer + AudioClassifier → fusion(concat) → logits

All outputs → temperature scaling (learned scalar) → final logits
```

### ImageDetector (EfficientNet-B4 Multi-Stream)
- **RGB stream**: EfficientNet-B4 backbone (1792-dim features, pretrained ImageNet)
- **Frequency stream**: 1-channel FFT magnitude → 2-layer CNN → 32-dim
- **Edge stream**: 1-channel Canny edge map → 2-layer CNN → 32-dim
- **Fusion**: concat (1792 + 32 + 32 = 1856) → Dropout → Linear(1856→512) → Linear(512→3)
- **Returns**: (logits, combined_features) — features reused by VideoAnalyzer

### VideoAnalyzer (Temporal Transformer)
- Reuses `ImageDetector` as frame-level feature extractor (no separate backbone)
- Frame features (batch, seq_len, 1856) → `TransformerEncoder` (3 layers, 8 heads, d_model=1856)
- Global Average Pooling over time → 1856-dim video feature
- Classifier: Linear(1856→256) → ReLU → Dropout(0.4) → Linear(256→3)

### AudioClassifier (CNN-LSTM Hybrid)
- **Input**: Mel-spectrogram (1, 128, time) + 8 auxiliary statistical features
- **CNN**: 3×(Conv2d → BatchNorm → ReLU → MaxPool2d) → 64 channels
- **LSTM**: 2-layer bidirectional over time frames → 128-dim hidden state
- **Aux Dense**: Linear(8→32)
- **Fusion**: concat (128 + 32 = 160) → Dropout → Linear(160→64) → Linear(64→3)

### 3-Class Output
All models output logits for 3 classes:
- `0 = REAL`
- `1 = MANIPULATED` (e.g., FaceSwap, morphed)
- `2 = AI_GENERATED` (e.g., Midjourney, Stable Diffusion)

---

## Training Architecture

### deepfake_live_detector Training Pipeline (`training/train.py`)
1. **`prepare_datasets()`** — crawl `d:\VIT\Datasets` for image/audio files, write list files
2. **`ListDataset`** — reads `{mode}_train_list.txt` (format: `<path> <label_int>`)
3. **`train_model()`** — AdamW optimizer, FocalLoss (γ=2), CosineAnnealing LR, mixed-precision (CUDA only via `torch.amp.GradScaler`)
4. **`evaluate_model()`** — per-class accuracy on test split
5. Saves best checkpoint + final checkpoint to `weights/`

### deepfake_detector Training (`training/train_ensemble.py`)
- Separate ensemble training; dataset loaded via `dataset_loader.py`
- Jupyter notebook `deepfake_training.ipynb` provides interactive training

---

## Entry Points

| Command | Module | Entry Point |
|---|---|---|
| `uvicorn inference.api_server:app --port 8000` | deepfake_detector | FastAPI REST API |
| `streamlit run app.py --server.port 8501` | deepfake_detector | Streamlit UI (calls API) |
| `streamlit run frontend/streamlit_app.py --server.port 8501` | deepfake_live_detector | Live Streamlit dashboard |
| `python training/train.py --mode image` | deepfake_live_detector | Training script |
| `python export_onnx.py` | deepfake_live_detector | ONNX model export |
