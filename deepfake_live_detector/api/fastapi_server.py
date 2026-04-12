"""FastAPI server for deepfake detection — image + audio endpoints."""
import os, sys, time, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="Deepfake Live Detector API", version="2.0.0",
              description="Real-time deepfake detection for images and audio.")

from realtime.live_inference import predict_image, predict_audio

TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "heic", "bmp", "tiff"}
AUDIO_EXTS = {"mp3", "wav", "m4a", "flac", "ogg"}


class DetectionResult(BaseModel):
    filename: str
    file_type: str
    source: str
    prediction: str
    confidence: float
    anomalies: List[str]
    reasons: List[str]
    detection_time_ms: int


@app.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    tmp_path = TMP / file.filename
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if ext in IMAGE_EXTS:
            res = predict_image(str(tmp_path))
            ftype = "image"
        elif ext in AUDIO_EXTS:
            res = predict_audio(str(tmp_path))
            ftype = "audio"
        else:
            raise HTTPException(400, f"Unsupported format: .{ext}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return DetectionResult(
        filename=file.filename, file_type=ftype, source="file_upload", **res,
    )


@app.get("/health")
def health():
    return {"status": "ok", "gpu": __import__("torch").cuda.is_available()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
