"""Stage 10: assemble the final explainable forensic report from all prior stages.

Combines hard signals (provenance/manifest), the learned-detector softmax, quality
flags, and classical-forensics anomalies into the structured report schema used by
the API: classification, confidence, alternatives, localized_regions, evidence, limitations.
"""


def finalize_report(stages: dict) -> dict:
    """Merge per-stage results into a final verdict.

    stages: dict with file_info, provenance, metadata, quality, classical_forensics,
    detector (softmax + anomalies), robustness, localized_regions.
    """
    s2 = stages.get("provenance") or {}
    s3 = stages.get("metadata") or {}
    s4 = stages.get("quality") or {}
    s5 = stages.get("classical_forensics") or {}
    det = stages.get("detector") or {}

    report = {
        "classification": "Inconclusive",
        "confidence": 0.0,
        "alternatives": {},
        "localized_regions": stages.get("localized_regions", []),
        "evidence": [],
        "limitations": [],
    }

    probs = det.get("probabilities") or {}
    if probs:
        report["alternatives"] = {str(k): round(float(v), 4) for k, v in probs.items()}

    # 1. Hard provenance signal overrides the learned verdict.
    if s2.get("c2pa_manifest_found") and s2.get("is_valid"):
        report["classification"] = "REAL"
        report["confidence"] = round(max(0.6, probs.get("REAL", 0.5)), 4)
        report["evidence"].append(
            "Valid C2PA provenance manifest cryptographically bound to the file.")
        return report
    if s2.get("c2pa_manifest_found") and not s2.get("is_valid"):
        report["evidence"].append("C2PA manifest found but validation failed or is incomplete.")
        report["limitations"].append("Invalid or unsigned provenance manifest present.")

    # 2. Learned detector is the primary signal.
    if probs:
        cls = max(probs, key=probs.get)
        conf = probs[cls]
        if conf < 0.55:
            report["classification"] = "Inconclusive"
            report["limitations"].append(f"Detector confidence {conf:.2f} below 0.55 threshold.")
            report["confidence"] = round(conf, 4)
        else:
            report["classification"] = cls
            report["confidence"] = round(conf, 4)
            report["evidence"].append(
                f"Learned ensemble classified the image as {cls} with {conf:.1%} confidence.")

    # 3. Explicit AI-generation metadata tags.
    if s3.get("has_ai_tags"):
        report["evidence"].append(
            f"AI-generation markers found in metadata (software: {s3.get('software')}).")
        if report["classification"] == "REAL":
            report["classification"] = "FAKE"
            report["confidence"] = max(report["confidence"], 0.7)
    elif report["classification"] == "REAL":
        report["evidence"].append("No AI-generation metadata markers detected.")

    # 4. Quality and classical forensics corroborate/adjust.
    if s4.get("blur_score", 1000) < 100:
        report["limitations"].append(
            "Image is blurry (low Laplacian variance); forensic signals may be degraded.")
    if s4.get("is_image", False) and not s4.get("width", 0):
        report["limitations"].append("Image dimensions could not be read.")

    if s5.get("double_compression_suspected"):
        report["evidence"].append("Double JPEG-compression artifacts suspected.")
    if s5.get("cfa_artifacts_detected"):
        report["evidence"].append("Color-filter-array interpolation inconsistencies detected.")

    if report["classification"] == "Inconclusive" and report["limitations"]:
        report["confidence"] = round(max(0.4, report["confidence"]), 4)

    # 5. Robustness feedback.
    rob = stages.get("robustness") or {}
    if rob.get("unstable"):
        report["limitations"].append(
            "Prediction changed after JPEG re-encoding — result may be unreliable.")

    for lim in det.get("limitations", []):
        if lim not in report["limitations"]:
            report["limitations"].append(lim)

    return report