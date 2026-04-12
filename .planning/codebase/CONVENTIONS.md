# CONVENTIONS.md — Code Style & Patterns

## Language & Style

- **Python** throughout; no formal linting config found (no `.flake8`, `.pylintrc`, `pyproject.toml` with ruff/black settings)
- **Docstrings**: triple-quoted `"""` strings used consistently for module-level and class-level docs; function docstrings are present on critical utilities but sparse in model files
- **Comments**: inline comments used liberally; section separators use ASCII art dashes:
  ```python
  # ----------------- IMAGE FEATURE EXTRACTORS -----------------
  # ─── Audio Feature Analysis for Reasoning ────────────────────────────
  ```
- **Line length**: no enforced limit observed; some lines are long (80–120+ chars)
- **Imports**: standard library first, then third-party, then local — loose adherence; `sys.path.insert()` hacks used to enable cross-directory imports

## Naming Conventions

| Category | Style | Example |
|---|---|---|
| Files | `snake_case` | `live_inference.py`, `feature_extractors.py` |
| Classes | `PascalCase` | `DeepfakeEnsemble`, `ListDataset`, `FocalLoss` |
| Functions | `snake_case` | `predict_image()`, `extract_audio_features()` |
| Private helpers | `_snake_case` | `_get_device()`, `_load_image_model()`, `_analyze_image_features()` |
| Constants | `UPPER_CASE` | `CLASSES`, `WEIGHTS`, `ROOT`, `API_URL` |
| Module singletons | `_lower` | `_device`, `_img_model`, `_aud_model` |
| Class members | `snake_case` | `self.backbone`, `self.freq_head`, `self.temperature` |

## PyTorch Conventions

### Model Definition Pattern
All models follow the `nn.Module` pattern:
```python
class MyModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MyModel, self).__init__()
        self.backbone = ...
        self.classifier = nn.Sequential(...)

    def forward(self, x, aux=None):
        ...
        return out, features  # always return both logits AND features
```

### Dual Return Convention
All sub-models return `(logits, features)` to allow ensemble reuse without re-forward:
```python
# VideoAnalyzer reuses ImageDetector features
_, frame_features = self.frame_extractor(frames_flat, freqs_flat, edges_flat)
```

### Device Handling
Device detection always uses the same priority pattern:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# or with MPS (live_detector):
if torch.cuda.is_available(): _device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): ...
else: _device = torch.device("cpu")
```

### Inference Mode
All inference paths use `torch.no_grad()` context manager:
```python
with torch.no_grad():
    out = model(...)
    probs = torch.nn.functional.softmax(out, dim=1).squeeze()
```

### Model Loading
Weights loaded with `map_location=device` for cross-platform compatibility:
```python
model.load_state_dict(torch.load(weight_path, map_location=device))
model.eval()
```

## Error Handling Patterns

### API Error Handling (`inference/api_server.py`)
```python
try:
    pred, conf, anomalies = infer_image(file_location)
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
finally:
    if os.path.exists(file_location):
        os.remove(file_location)  # Always cleanup temp files
```

### Training Error Recovery (`training/train.py`)
Batches that fail don't crash the epoch; they log and continue:
```python
except Exception as e:
    print(f"\n  [ERROR] Batch {i+1} failed: {e}", flush=True)
    traceback.print_exc()
    # continues to next batch
```

### Dataset Loading Resilience (`training/train.py` → `ListDataset.__getitem__`)
Returns zero tensors on corrupt files instead of crashing:
```python
except Exception as e:
    return torch.zeros(3, 512, 512), torch.zeros(1, 512, 512), torch.zeros(1, 512, 512), label
```

### Streamlit User Error Display
Uses Streamlit's own `st.error()` / `st.warning()` / `st.success()` for user-facing messages; `requests.exceptions.ConnectionError` specifically caught to give a helpful "API not running" message.

## Feature Extraction Patterns

### Multi-Modal Input Tuple Convention
Feature extractors always return **tuples** for multi-stream models:
```python
# Image: rgb_tensor, fft_tensor, edge_tensor
rgb, freq, edge = preprocess_image(filepath)
model(image_tuple=(rgb.unsqueeze(0), freq.unsqueeze(0), edge.unsqueeze(0)))

# Audio: mel_spectrogram_tensor, aux_features_tensor
mel, aux = extract_audio_features(audio_path)
model(audio_tuple=(mel, aux.unsqueeze(0)))
```

### Graceful Degradation with Zeros
When optional auxiliary inputs are missing, models substitute zeros:
```python
if freq is not None:
    freq_feat = self.freq_head(freq)
    features.append(freq_feat)
else:
    features.append(torch.zeros(rgb.shape[0], 32, device=rgb.device))
```

## Streamlit UI Patterns

### Custom CSS via `st.markdown(unsafe_allow_html=True)`
The live_detector UI uses a large embedded `<style>` block with:
- Dark glassmorphism theme (dark navy + blur + semi-transparent cards)
- CSS variables for consistent color tokens
- Google Fonts (Inter) imported via CDN

### Session State for History
```python
if "history" not in st.session_state:
    st.session_state.history = []
```

### Tab-based Layout
`st.tabs(["📷 Camera", "🎙️ Microphone", "📁 Upload", "📊 History"])` for organized multi-mode UI

## Print Logging

Training uses `print(..., flush=True)` throughout for real-time log visibility in containers/terminals. No Python `logging` module used.
