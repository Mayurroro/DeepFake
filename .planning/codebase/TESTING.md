# TESTING.md — Test Structure & Practices

## Current State

**No automated test suite exists** in either module. There are no:
- `tests/` or `test_*/` directories
- `pytest`, `unittest`, or any test framework configured
- CI/CD configuration files (no `.github/workflows/`, no `Makefile` with test targets)
- `conftest.py`, `pytest.ini`, `setup.cfg [tool:pytest]`

---

## What Functions as Testing

### 1. Debug/Validation Scripts

**`deepfake_live_detector/debug_train.py`** (1.6 KB)
- Manual debug helper for training issues
- Not a test framework; run manually to diagnose training crashes

**`deepfake_live_detector/training/run_training.py`** (0.6 KB)
- Simple launcher for `train.py`; acts as a smoke test for training pipeline

### 2. Training Evaluation (Informal)

`training/train.py` includes an `evaluate_model()` function that reports:
- Overall test accuracy (`correct/total`)
- Per-class breakdown (REAL / MANIPULATED / AI_GENERATED)

This is the closest thing to automated validation:
```python
acc = correct / total if total > 0 else 0
print(f"\n  Test Accuracy: {acc:.2%} ({correct}/{total})")
for cls_idx, cls_name in enumerate(classes):
    cls_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == cls_idx and p == cls_idx)
    print(f"  {cls_name}: {cls_correct}/{cls_total} ({cls_correct / cls_total:.2%})")
```

### 3. Jupyter Notebook Exploration

**`deepfake_detector/deepfake_training.ipynb`** serves as exploratory analysis:
- Interactive training runs with visual feedback
- Not automated; requires manual execution

### 4. API Smoke Test

`deepfake_detector/app.py` (Streamlit) doubles as an integration test:
- Upload a file and press "Run Detection Models"
- Verifies Streamlit → FastAPI → Model → Response pipeline end-to-end

---

## How to Manually Test

### End-to-End: deepfake_detector
```bash
# Terminal 1 — Start API
cd deepfake_detector
uvicorn inference.api_server:app --port 8000

# Terminal 2 — Start UI
streamlit run app.py --server.port 8501
# Open http://localhost:8501, upload an image/audio, click "Run Detection Models"
```

### End-to-End: deepfake_live_detector
```bash
cd deepfake_live_detector
streamlit run frontend/streamlit_app.py --server.port 8501
# Camera tab: capture webcam frame → verify result card
# Mic tab: record audio → verify result card
# Upload tab: drag & drop image or audio → verify result card
```

### Direct Model Inference Test
```python
from realtime.live_inference import predict_image, predict_audio
result = predict_image("path/to/test_image.jpg")
print(result)  # {"prediction": "REAL", "confidence": 0.97, "anomalies": [], ...}
```

### Training Pipeline Test
```bash
cd deepfake_live_detector
python training/train.py --mode image --epochs 1 --batch 8
# Verifies: dataset loading, model init, forward pass, loss, checkpoint save
```

---

## Recommendations for Future Test Suite

### Suggested Structure
```
tests/
├── unit/
│   ├── test_feature_extractors.py   # Test FFT, edge, audio feature extraction
│   ├── test_models.py               # Test forward pass shape/dtype
│   └── test_inference.py            # Test predict_image / predict_audio contracts
├── integration/
│   ├── test_api.py                  # FastAPI TestClient POST /detect
│   └── test_training.py             # 1-epoch smoke test on synthetic data
└── conftest.py                      # Fixtures: dummy images, dummy audio
```

### Key Tests to Add First
1. **Feature extractor shape tests** — `preprocess_image()` returns correct tensor shapes
2. **Model forward pass** — `ImageDetector(pretrained=False)(dummy_rgb, dummy_freq, dummy_edge)` runs without error and returns (3,) logits
3. **API contract** — `POST /detect` with a real image returns `DetectionResponse` schema
4. **Dataset loader** — `ListDataset` handles missing files without crash
5. **Inference output contract** — `predict_image()` always returns dict with required keys

### Suggested Framework
```bash
pip install pytest pytest-mock httpx
# httpx for FastAPI async test client
```
```python
# Example model shape test
def test_image_detector_output_shape():
    from models.image_detector import ImageDetector
    model = ImageDetector(pretrained=False)
    rgb = torch.randn(2, 3, 512, 512)
    freq = torch.randn(2, 1, 512, 512)
    edge = torch.randn(2, 1, 512, 512)
    logits, features = model(rgb, freq, edge)
    assert logits.shape == (2, 3)
    assert features.shape == (2, 1856)
```
