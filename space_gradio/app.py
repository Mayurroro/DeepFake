import spaces  # MUST stay first import on ZeroGPU

import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))

IMAGE_CLASSES = ["REAL", "FAKE"]
AUDIO_CLASSES = ["REAL", "MANIPULATED", "AI_GENERATED"]
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "heic"}
AUDIO_EXTS = {"wav", "flac", "ogg", "mp3", "m4a"}


def _session(model_file):
    """Load an ONNX session once (read-only afterward — safe under concurrency)."""
    return ort.InferenceSession(
        os.path.join(HERE, "onnx_models", model_file),
        providers=["CPUExecutionProvider"],
    )


_image_sess = _session("image_detector.onnx")
_audio_sess = _session("audio_detector.onnx")


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _preprocess_image(path):
    """Mirror of utils/feature_extractors.preprocess_image_ensemble (numpy)."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = 20 * np.log(np.abs(f) + 1e-8)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    gray_f = gray.astype(np.float32) / 255.0
    resid = gray_f - cv2.GaussianBlur(gray_f, (5, 5), 0)
    resid = (resid - resid.mean()) / (resid.std() + 1e-8)

    rgb = img.astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    rgb = ((rgb - mean) / std).astype(np.float32)

    return {"rgb": rgb[None], "noise": np.stack([resid] * 3, axis=-1).transpose(2, 0, 1).astype(np.float32)[None],
            "freq": mag.astype(np.float32)[None, None]}


def _preprocess_audio(path, sr=16000, max_time_steps=400):
    """Mirror of utils/feature_extractors.extract_audio_features (numpy)."""
    import librosa
    y, sr = librosa.load(path, sr=sr)

    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000), ref=np.max)
    if S.shape[1] > max_time_steps:
        S = S[:, :max_time_steps]
    else:
        S = np.pad(S, ((0, 0), (0, max_time_steps - S.shape[1])))

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    aux = np.concatenate([
        [np.mean(cent), np.var(cent)],
        [np.mean(roll), np.var(roll)],
        np.mean(mfcc, axis=1),
        np.var(mfcc, axis=1),
    ])
    return {"mel": S.astype(np.float32)[None, None], "aux": aux.astype(np.float32)[None]}


@spaces.GPU(duration=10)
def gpu_probe():
    """Probe the ZeroGPU slot (verification only; the demo itself runs on CPU via ONNX)."""
    import torch
    return f"GPU slot OK — torch {torch.__version__}, cuda={torch.cuda.is_available()}"


def analyze(file):
    """Run deepfake detection on an uploaded image or audio file.

    Args:
        file: path to an uploaded image or audio file.
    Returns:
        verdict: class confidence distribution as a dict, plus detail string.
    """
    if file is None:
        raise gr.Error("Upload an image or audio file first.")
    ext = os.path.splitext(file.name)[-1].lower().lstrip(".")
    t0 = time.perf_counter()

    if ext in IMAGE_EXTS:
        logits = _image_sess.run(["logits"], _preprocess_image(file.name))[0][0]
        classes, kind = IMAGE_CLASSES, "image"
    elif ext in AUDIO_EXTS:
        logits = _audio_sess.run(["logits"], _preprocess_audio(file.name))[0][0]
        classes, kind = AUDIO_CLASSES, "audio"
    else:
        raise gr.Error(f"Unsupported file type: .{ext}")

    probs = _softmax(logits)
    probs = {c: round(float(p), 4) for c, p in zip(classes, probs)}
    verdict = max(probs, key=probs.get)
    ms = int((time.perf_counter() - t0) * 1000)

    detail = (
        f"### Verdict: **{verdict}**\n\n"
        f"File type: {kind}  ·  Inference: **{ms} ms** on CPU (ONNX Runtime)\n\n"
        "Beware: model scores are not calibrated truth. Try flipping the picture, "
        "re-encoding audio, or testing cropped regions — a single score is opinion, not proof."
    )
    return probs, detail


with gr.Blocks(title="Deepfake Live Detector") as demo:
    gr.Markdown(
        "# 🛡️ Deepfake Live Detector\n"
        "EfficientNet-B4 + FFT/edge noise branches (image) and CNN-LSTM-MHA (audio) "
        "trained on image & audio deepfakes. Runs fully on **CPU via ONNX Runtime** — "
        "this demo never requests a GPU."
    )
    with gr.Row():
        upload = gr.File(label="Upload image or audio", file_types=None)
        run = gr.Button("Detect", variant="primary")
    probs_out = gr.Label(label="Class probabilities")
    detail_out = gr.Markdown()
    with gr.Accordion("ZeroGPU diagnostics", open=False):
        probe_btn = gr.Button("Check GPU slot")
        probe_out = gr.Text()
        probe_btn.click(gpu_probe, inputs=[], outputs=probe_out)
    run.click(analyze, inputs=[upload], outputs=[probs_out, detail_out])

demo.launch()