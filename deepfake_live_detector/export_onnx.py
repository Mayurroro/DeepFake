"""Export trained models to ONNX for edge deployment."""
import sys, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

WEIGHTS = ROOT / "weights"
OUT = ROOT / "onnx_models"
OUT.mkdir(exist_ok=True)


def export_image():
    from models.image_detector import ImageDetector
    m = ImageDetector(pretrained=False).eval()
    w = WEIGHTS / "image_detector.pth"
    if w.exists():
        m.load_state_dict(torch.load(w, map_location="cpu"))
    dummy = (torch.randn(1, 3, 512, 512), torch.randn(1, 1, 512, 512), torch.randn(1, 1, 512, 512))
    out = OUT / "image_detector.onnx"
    torch.onnx.export(m, dummy, str(out), input_names=["rgb", "freq", "edges"],
                      output_names=["logits", "features"], opset_version=14)
    print(f"✓ Image detector → {out}")


def export_audio():
    from models.audio_detector import AudioDetector
    m = AudioDetector().eval()
    w = WEIGHTS / "audio_detector.pth"
    if w.exists():
        m.load_state_dict(torch.load(w, map_location="cpu"))
    dummy = (torch.randn(1, 1, 128, 400), torch.randn(1, 30))
    out = OUT / "audio_detector.onnx"
    torch.onnx.export(m, dummy, str(out), input_names=["mel", "aux"],
                      output_names=["logits", "features"], opset_version=14)
    print(f"✓ Audio detector → {out}")


if __name__ == "__main__":
    export_image()
    export_audio()
    print("Done! ONNX models saved in:", OUT)
