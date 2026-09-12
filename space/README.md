---
title: Deepfake Live Detector
emoji: 🛡️
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
license: apache-2.0
---

**Deepfake Live Detector** — in-browser deepfake image detection.

All inference happens locally in your browser via ONNX Runtime Web (WASM) — no server, no GPU, no upload.
Image model: EfficientNet-B4 multi-branch ensemble (RGB / FFT spectrum / noise-residual).

Audio detection and the full pipeline are in the [repository](https://github.com/) (Gradio app, swap-in when the
Space is upgraded to run compute).