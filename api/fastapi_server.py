"""FastAPI server for deepfake detection — image + audio endpoints."""
import os, sys, time, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="Deepfake Live Detector API", version="2.0.0",
              description="Real-time deepfake detection for images and audio.")

from realtime.live_inference import predict_audio
from utils.pipeline import run_forensic_pipeline

TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(ROOT / "frontend" / "index.html"))

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "heic", "bmp", "tiff"}
AUDIO_EXTS = {"mp3", "wav", "m4a", "flac", "ogg"}


class LocalizedRegion(BaseModel):
    region: str
    operation: str
    confidence: float

class ForensicReport(BaseModel):
    filename: str
    file_type: str
    classification: str
    confidence: float
    alternatives: Dict[str, float] = {}
    localized_regions: List[LocalizedRegion] = []
    evidence: List[str] = []
    limitations: List[str] = []
    file_info: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None
    classical_forensics: Optional[Dict[str, Any]] = None
    detector: Optional[Dict[str, Any]] = None
    robustness: Optional[Dict[str, Any]] = None


@app.post("/detect", response_model=ForensicReport)
async def detect(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    tmp_path = TMP / file.filename
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if ext in IMAGE_EXTS:
            report_data = run_forensic_pipeline(str(tmp_path))
            ftype = "image"
            
            # Construct report from pipeline
            return ForensicReport(
                filename=file.filename,
                file_type=ftype,
                **report_data
            )
            
        elif ext in AUDIO_EXTS:
            # Fallback wrapper to fit ForensicReport schema until audio is updated
            res = predict_audio(str(tmp_path))
            ftype = "audio"
            return ForensicReport(
                filename=file.filename,
                file_type=ftype,
                classification=res.get("prediction", "Unknown"),
                confidence=res.get("confidence", 0.0),
                evidence=res.get("reasons", []),
                limitations=res.get("anomalies", [])
            )
        else:
            raise HTTPException(400, f"Unsupported format: .{ext}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "gpu": __import__("torch").cuda.is_available()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
