# CONCERNS.md — Technical Debt, Known Issues & Areas of Concern

## 🔴 Critical / High Priority

### 1. Video Inference is a Stub — Not Functional
**File:** `deepfake_detector/inference/api_server.py` (lines 44–52)

The `infer_video()` function is entirely hardcoded and non-functional:
```python
def infer_video(filepath):
    frames = extract_frames(filepath, fps=5)
    if not frames:
        raise ValueError("Video contains no readable frames")
    # Preprocess frames and run through model here...
    time.sleep(1.5)  # Simulate processing
    return "AI_GENERATED", 0.965, ["motion_physics_violation", "frame_flickering"]
```
**Impact:** Any video uploaded to `deepfake_detector` gets a hardcoded fake result regardless of content. This is a fundamental gap in the system's claimed capabilities.

---

### 2. Nested `.git` Repository in deepfake_detector
**Location:** `deepfake_detector/.git/`

`deepfake_detector` has its own `.git` directory inside the parent repository. This means it was either an independent repo or initialized separately. This is a Git submodule situation without proper submodule configuration — it will cause confusing behavior when the parent repo tries to track files in `deepfake_detector`.  
**Risk:** `git add .` from root won't track changes inside `deepfake_detector/` correctly.

---

### 3. No Authentication on FastAPI `/detect` Endpoint
**File:** `deepfake_detector/inference/api_server.py` (line 65)

```python
@app.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(file: UploadFile = File(...)):
```
No auth headers, API keys, or rate limiting. Any process that can reach port 8000 can submit arbitrary files.  
**Risk:** Unauthenticated file uploads; potential for DoS via large file bombing or malicious binaries.

---

### 4. No Model Loaded in API Server by Default
**File:** `deepfake_detector/inference/api_server.py` (lines 20–22)
```python
weight_path = "../weights/deepfake_model_latest.pth"
if os.path.exists(weight_path):
    model.load_state_dict(...)
model.eval()
```
If weights don't exist (which they won't in a fresh clone), the model runs with **random untrained weights**. There is no error or warning to the user — predictions appear normal but are meaningless.

---

## 🟡 Medium Priority

### 5. Duplicate `feature_extractors.py` / `image_detector.py` Across Modules
Both `deepfake_detector/utils/feature_extractors.py` and `deepfake_live_detector/utils/feature_extractors.py` contain largely identical code with slight differences. Similarly, `models/image_detector.py` exists in both modules.  
**Risk:** Bug fixes applied to one module won't propagate to the other. DRY violation.

### 6. Path Hacks via `sys.path.insert()`
Multiple files use `sys.path.insert(0, str(ROOT))` to enable cross-directory imports:
```python
# live_inference.py
import sys; sys.path.insert(0, str(ROOT))
from models.image_detector import ImageDetector
```
This is fragile — it depends on working directory and file location at runtime. Packaging conflicts could arise.

### 7. Hardcoded Dataset Path
**File:** `deepfake_live_detector/training/prepare_datasets.py`

The dataset root is hardcoded as `d:\VIT\Datasets`. This will fail on any machine or OS where the path differs.  
**Fix:** Use environment variable, `argparse`, or config file for dataset path.

### 8. Temp File Cleanup Not Guaranteed in deepfake_detector
**File:** `deepfake_detector/inference/api_server.py` (line 96)

Cleanup is inside a `finally` block, but if the server process is killed mid-request, the temp file may remain in `../tmp/`.  
**Fix:** Use `tempfile.NamedTemporaryFile` with `delete=True` or a proper temp directory per request.

### 9. Relative Path for Weight Loading in deepfake_detector API
```python
weight_path = "../weights/deepfake_model_latest.pth"
```
This is relative to the process working directory, not the file's location. Will fail if `uvicorn` is launched from a different directory.  
**Fix:** Use `Path(__file__).resolve().parent.parent / "weights" / "deepfake_model_latest.pth"`.

### 10. `audio_test_list.txt` is Empty in deepfake_detector
**File:** `deepfake_detector/training/audio_test_list.txt` (0 bytes)

The audio test list is completely empty, meaning there's no audio test split for the batch detector module.

---

## 🟢 Low Priority / Observations

### 11. No Logging Framework
All logging is done via `print(..., flush=True)`. This lacks log levels, timestamps, structured output, or file logging. Makes production debugging harder.  
**Fix:** Replace with Python `logging` module or `loguru`.

### 12. Missing Pre/Post Processing for Batch Video Inference (deepfake_detector)
`VideoAnalyzer` model exists but `infer_video()` in the API never actually calls it. The FFT/edge preprocessing pipeline for video frames is also missing (would need to be applied to each extracted frame before passing to the model).

### 13. ONNX Export Not Wired into Inference
`deepfake_live_detector/export_onnx.py` exists and can export models, but `onnxruntime` is never used in the active inference path. The ONNX export is a dead-end upgrade path.

### 14. streamlit-webrtc Dependency Not Used
`requirements.txt` lists `streamlit-webrtc>=0.47.0` but `frontend/streamlit_app.py` uses Streamlit's native `st.camera_input()` and `st.audio_input()` instead of WebRTC. Unnecessary dependency that adds complexity to install on environments without browser WebRTC support (especially Docker).

### 15. Aux Feature Size Mismatch Risk in AudioClassifier
`AudioClassifier` expects 8-dimensional `aux_features` (based on the dense layer `nn.Linear(8, 32)`), but `extract_audio_features()` constructs aux_features as:
```python
aux_features = np.concatenate([
    [mean_centroid, var_centroid],   # 2
    [mean_rolloff, var_rolloff],     # 2
    np.mean(mfccs, axis=1),          # 13 (n_mfcc=13)
    np.var(mfccs, axis=1)            # 13 (n_mfcc=13)
])  # Total: 30 features
```
This produces a **30-dimensional** vector, not 8. This would cause a runtime dimension mismatch error on actual audio inference. The comment in `AudioClassifier` says "8 statistical features extracted via librosa" but the actual extractor produces 30. This is a **latent runtime bug**.

### 16. No Input Validation / File Size Limits
The API accepts any file upload with no size limit. Large video files could crash the server or exhaust disk space in `../tmp/`.

### 17. No Versioning or API Versioning
FastAPI app is `version="1.0.0"` but there's no actual version enforcement or `/v1/` route prefix. Breaking changes could silently break the Streamlit client.
