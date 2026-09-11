import os
import cv2
import numpy as np

from .stage1_file_inspection import inspect_file
from .stage2_provenance import validate_provenance
from .stage3_metadata import extract_metadata
from .stage4_quality import assess_quality
from .stage5_classical import extract_classical_features
from .stage10_report import finalize_report


def _localize_regions(img_bgr, top_k=2):
    """Heuristic manipulation localization: grid cells with the highest noise-residual variance."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    resid = gray_f - cv2.GaussianBlur(gray_f, (5, 5), 0)
    h, w = resid.shape
    gy = gx = 4
    cells = []
    for i in range(gy):
        for j in range(gx):
            cell = resid[i*h//gy:(i+1)*h//gy, j*w//gx:(j+1)*w//gx]
            cells.append(((j, i), float(cell.var())))
    cells.sort(key=lambda c: c[1], reverse=True)
    regions = []
    for (j, i), var in cells[:top_k]:
        if var > 1e-4:
            regions.append({
                "region": f"quadrant_{j}x{i}",
                "operation": "noise_residual_anomaly",
                "confidence": round(min(1.0, var * 1e3), 3),
            })
    return regions


def _run_detector(file_path, img_bgr):
    """Stages 6-7: learned multi-branch detector + gated evidence fusion."""
    from realtime.live_inference import predict_image
    try:
        res = predict_image(file_path)
    except Exception as e:
        return {"error": str(e), "probabilities": {}, "anomalies": [], "reasons": [], "limitations": []}
    probs = {c: res.get("probabilities", {}).get(c, 0.0) for c in ("REAL", "FAKE")}
    return {
        "prediction": res.get("prediction"),
        "confidence": res.get("confidence"),
        "probabilities": res.get("probabilities", {}),
        "anomalies": res.get("anomalies", []),
        "reasons": res.get("reasons", []),
        "detection_time_ms": res.get("detection_time_ms"),
        "limitations": res.get("limitations", []),
    }


def _check_robustness(img_bgr, original_prediction):
    """Stage 8: robustness — does the verdict survive JPEG re-encoding?"""
    from realtime.live_inference import predict_image
    ok, enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return {"unstable": False, "note": "re-encode failed"}
    re_encoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    try:
        res = predict_image(re_encoded)
        return {
            "unstable": res.get("prediction") != original_prediction,
            "reencoded_prediction": res.get("prediction"),
        }
    except Exception:
        return {"unstable": False, "note": "re-encode inference failed"}


def run_forensic_pipeline(file_path: str) -> dict:
    """
    Orchestrates the 10-stage hybrid forensic pipeline.

    Stages 1-5: hard forensics (container, provenance, metadata, quality, classical).
    Stages 6-7: learned ensemble (multi-branch features + gated fusion).
    Stage 8:    robustness under recompression.
    Stage 9-10: final verdict assembly into the explainable report.
    """
    report = {
        "classification": "Inconclusive",
        "confidence": 0.0,
        "alternatives": {},
        "localized_regions": [],
        "evidence": [],
        "limitations": []
    }

    if not os.path.exists(file_path):
        report["limitations"].append("File not found")
        return report

    # Stage 1: File Inspection
    s1_res = inspect_file(file_path)
    report["file_info"] = s1_res

    # Stage 2: Provenance
    s2_res = validate_provenance(file_path)
    report["provenance"] = s2_res

    # Stage 3: Metadata
    s3_res = extract_metadata(file_path)
    report["metadata"] = s3_res

    # Stage 4: Quality Assessment
    s4_res = assess_quality(file_path)
    report["quality"] = s4_res

    # Stage 5: Classical Forensics
    s5_res = extract_classical_features(file_path)
    report["classical_forensics"] = s5_res

    stages = {
        "file_info": s1_res,
        "provenance": s2_res,
        "metadata": s3_res,
        "quality": s4_res,
        "classical_forensics": s5_res,
    }

    # Stages 6-7: Learned detector + fusion (runs on any decodable image)
    img = cv2.imread(file_path)
    if img is not None:
        det_res = _run_detector(file_path, img)
        report["detector"] = det_res
        stages["detector"] = det_res

        # Stage 8: Robustness
        rob_res = _check_robustness(img, det_res.get("prediction"))
        report["robustness"] = rob_res
        stages["robustness"] = rob_res

        # Localization for suspicious images
        if det_res.get("prediction") != "REAL":
            report["localized_regions"] = _localize_regions(img)
            stages["localized_regions"] = report["localized_regions"]
    else:
        report["detector"] = {"error": "image could not be decoded"}
        stages["detector"] = report["detector"]

    # Stages 9-10: Final verdict
    final = finalize_report(stages)
    final["file_info"] = s1_res
    final["provenance"] = s2_res
    final["metadata"] = s3_res
    final["quality"] = s4_res
    final["classical_forensics"] = s5_res
    final["detector"] = report["detector"]
    final["robustness"] = report.get("robustness", {})
    return final