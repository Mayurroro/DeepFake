# INTEGRATIONS.md — External Services & APIs

## APIs & Network Calls

### Internal REST API (FastAPI → Streamlit)
- **deepfake_detector** module has a decoupled architecture:
  - `deepfake_detector/app.py` (Streamlit UI) calls `http://localhost:8000/detect` via `requests.post()`
  - `deepfake_detector/inference/api_server.py` serves the `/detect` POST endpoint
- **deepfake_live_detector** module is **tightly coupled**: Streamlit calls inference functions directly (no HTTP hop)

### No External API Calls
- No third-party cloud AI APIs (no OpenAI, AWS Rekognition, Google Vision, etc.)
- No external data services, message queues, or event buses
- No authentication providers (no OAuth, JWT, API keys)

## Databases

- **No database** — stateless design; no SQL, NoSQL, or vector DB
- Detection history is stored in `st.session_state` (in-memory per Streamlit session, lost on reload)

## File System & Storage

### Temporary File Storage
- `deepfake_detector`: temp files written to `../tmp/` relative to the inference server working dir; cleaned up after each request
- `deepfake_live_detector`: temp files written to `{ROOT}/tmp/` (e.g., `mic_recording.wav`); deleted with `tmp_path.unlink(missing_ok=True)` after analysis

### Model Weights
- `deepfake_detector/weights/deepfake_model_latest.pth` — ensemble model weights (optional, model runs untrained if missing)
- `deepfake_live_detector/weights/image_detector.pth` — image model weights
- `deepfake_live_detector/weights/audio_detector.pth` — audio model weights
- Weights directory also used as output location after training

### Dataset Lists
- `deepfake_detector/training/image_train_list.txt` / `image_test_list.txt` — text files listing image paths + labels
- `deepfake_live_detector/training/audio_train_list.txt` / `audio_train_list.txt` — similar for audio
- Dataset root: `d:\VIT\Datasets` (hardcoded in `prepare_datasets.py`)

## Font & Web Assets (CDN)

- `deepfake_live_detector/frontend/streamlit_app.py` loads **Google Fonts** (Inter) via CDN:
  ```css
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
  ```
- This requires internet access when the Streamlit app starts

## Pretrained Model Hub (timm)

- EfficientNet-B4 backbone is loaded via `timm.create_model('efficientnet_b4', pretrained=True)`
- On first run, `timm` downloads pretrained weights from PyTorch Hub/Hugging Face
- Requires internet connection on first cold start; cached afterward

## EXIF / Metadata Analysis

- Uses Python's `PIL.ExifTags` to read embedded EXIF data in uploaded images
- Checks for AI generation markers: `midjourney`, `dall-e`, `stable diffusion`, `novelai`, `ai generated` in EXIF Software/Comment fields
- All done locally; no external API call

## Docker / Containerization

- Both modules can be packaged independently; no docker-compose or shared network config found
- `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` base image requires NVIDIA GPU runtime on Docker host

## Webhooks / Real-time Streams

- No webhooks configured
- `streamlit-webrtc` (listed in deps) provides WebRTC camera/mic stream capability but the actual implementation in `frontend/streamlit_app.py` uses Streamlit's built-in `st.camera_input()` and `st.audio_input()` (no WebRTC dependency at runtime)

## ONNX Runtime

- `export_onnx.py` in `deepfake_live_detector/` exports trained models to ONNX format
- `onnxruntime` enables inference without PyTorch (lighter deployment); not yet wired into the main inference path
