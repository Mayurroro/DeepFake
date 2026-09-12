"""Export trained models to ONNX for edge deployment.

Sizes match preprocess_image_ensemble (256) / extract_audio_features (400),
with dynamic batch and spatial dims so runtime inputs of any size are accepted.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

WEIGHTS = ROOT / "weights"
OUT = ROOT / "onnx_models"
OUT.mkdir(exist_ok=True)


def export_image():
    import torch
    from models.image_detector import ImageDetectorEnsemble
    m = ImageDetectorEnsemble(pretrained=False, num_classes=2).eval()
    w = WEIGHTS / "image_detector.pth"
    if w.exists():
        m.load_state_dict(torch.load(w, map_location="cpu"))
    dummy = (torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256), torch.randn(1, 1, 256, 256))
    out = OUT / "image_detector.onnx"
    torch.onnx.export(m, dummy, str(out), input_names=["rgb", "noise", "freq"],
                      output_names=["logits", "features"], opset_version=14,
                      dynamic_axes={"rgb": {0: "batch", 2: "h", 3: "w"},
                                    "noise": {0: "batch"},
                                    "freq": {0: "batch", 2: "h", 3: "w"}})
    check_image_runs(out)
    print(f"[OK] Image detector - {out}")


def export_audio():
    import torch
    from models.audio_detector import AudioDetector
    m = AudioDetector().eval()
    w = WEIGHTS / "audio_detector.pth"
    if w.exists():
        m.load_state_dict(torch.load(w, map_location="cpu"))
    dummy = (torch.randn(1, 1, 128, 400), torch.randn(1, 30))
    out = OUT / "audio_detector.onnx"
    torch.onnx.export(m, dummy, str(out), input_names=["mel", "aux"],
                      output_names=["logits", "features"], opset_version=14,
                      dynamic_axes={"mel": {0: "batch"}, "aux": {0: "batch"}})
    check_audio_runs(out)
    print(f"[OK] Audio detector - {out}")


def check_image_runs(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rgb = np.random.rand(1, 3, 256, 256).astype(np.float32)
    noise = np.random.rand(1, 3, 256, 256).astype(np.float32)
    freq = np.random.rand(1, 1, 256, 256).astype(np.float32)
    logits = sess.run(["logits"], {"rgb": rgb, "noise": noise, "freq": freq})[0]
    assert logits.shape == (1, 2), f"unexpected logits shape {logits.shape}"
    print(f"  image ONNX self-check ok - logits {logits.shape}")


def check_audio_runs(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    mel = np.random.rand(1, 1, 128, 400).astype(np.float32)
    aux = np.random.rand(1, 30).astype(np.float32)
    logits = sess.run(["logits"], {"mel": mel, "aux": aux})[0]
    assert logits.shape == (1, 3), f"unexpected logits shape {logits.shape}"
    print(f"  audio ONNX self-check ok - logits {logits.shape}")


if __name__ == "__main__":
    export_image()
    export_audio()
    print("Done! ONNX models saved in:", OUT)
