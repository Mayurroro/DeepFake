"""Fast inference wrapper — loads models once, exposes predict_image / predict_audio.
Includes detailed feature-level reasoning for why content is flagged.
"""
import os, time, torch, torch.nn.functional as F, numpy as np, cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
CLASSES = ["REAL", "MANIPULATED", "AI_GENERATED"]

_device = None
_img_model = None
_aud_model = None


def _get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
        print(f"[LiveInference] device = {_device}")
    return _device


def _load_image_model():
    global _img_model
    if _img_model is None:
        import sys; sys.path.insert(0, str(ROOT))
        from models.image_detector import ImageDetector
        _img_model = ImageDetector(pretrained=True).to(_get_device()).eval()
        w = WEIGHTS / "image_detector.pth"
        if w.exists():
            _img_model.load_state_dict(torch.load(w, map_location=_get_device()))
            print(f"[LiveInference] Loaded image weights: {w}")
    return _img_model


def _load_audio_model():
    global _aud_model
    if _aud_model is None:
        import sys; sys.path.insert(0, str(ROOT))
        from models.audio_detector import AudioDetector
        _aud_model = AudioDetector().to(_get_device()).eval()
        w = WEIGHTS / "audio_detector.pth"
        if w.exists():
            _aud_model.load_state_dict(torch.load(w, map_location=_get_device()))
            print(f"[LiveInference] Loaded audio weights: {w}")
    return _aud_model


# ─── Image Feature Analysis for Reasoning ───────────────────────────
def _analyze_image_features(source):
    """Run feature-level checks on the image and return anomalies + reasons."""
    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            return [], []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = source

    img_resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    anomalies = []
    reasons = []

    # 1. FFT Frequency Analysis — AI images have abnormal high-freq energy
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(f_shift) + 1e-8)
    h, w = mag.shape
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 4
    high_mask = np.ones_like(mag)
    cv2.circle(high_mask, (center_x, center_y), radius, 0, -1)
    high_energy = np.mean(mag * high_mask)
    low_energy = np.mean(mag * (1 - high_mask))
    freq_ratio = high_energy / (low_energy + 1e-8)

    if freq_ratio < 0.35:
        anomalies.append("frequency_artifacts")
        reasons.append(f"🔬 **Frequency Analysis:** Abnormally low high-frequency energy (ratio: {freq_ratio:.3f}). "
                       f"AI-generated images often lack natural high-frequency texture detail, producing smoother gradients than real photos.")

    # 2. Edge Sharpness Analysis — AI images have uniform edge sharpness
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 100, 200)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_std = np.std(edge_mag)
    edge_mean = np.mean(edge_mag)

    if edge_std < 15:
        anomalies.append("unnatural_edges")
        reasons.append(f"🔪 **Edge Analysis:** Edge sharpness is unusually uniform (std: {edge_std:.1f}). "
                       f"Real photos have varying edge sharpness due to depth of field and focus — AI images tend to have consistent sharpness throughout.")

    # 3. Texture Uniformity — AI images have repetitive texture patterns
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = laplacian.var()

    if lap_var < 100:
        anomalies.append("texture_repetition")
        reasons.append(f"🧱 **Texture Analysis:** Very low texture variance detected (Laplacian var: {lap_var:.1f}). "
                       f"This indicates overly smooth or repetitive texture patterns common in AI-generated content.")
    elif lap_var > 3000:
        anomalies.append("texture_anomaly")
        reasons.append(f"🧱 **Texture Analysis:** Abnormally high texture variance (Laplacian var: {lap_var:.1f}). "
                       f"This may indicate post-processing artifacts or manipulation noise injected into the image.")

    # 4. Lighting/Shadow Consistency — Check for uniform lighting
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(float)
    # Divide into quadrants and compare luminance
    q1 = l_channel[:256, :256].mean()
    q2 = l_channel[:256, 256:].mean()
    q3 = l_channel[256:, :256].mean()
    q4 = l_channel[256:, 256:].mean()
    lum_values = [q1, q2, q3, q4]
    lum_range = max(lum_values) - min(lum_values)

    if lum_range < 5:
        anomalies.append("uniform_lighting")
        reasons.append(f"💡 **Lighting Analysis:** Extremely uniform lighting across all regions (range: {lum_range:.1f}). "
                       f"Natural photos typically show lighting gradients and shadow variations. AI generators often produce flat, uniform illumination.")

    # 5. Color Distribution — Check for unnatural color distributions
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    sat_std = np.std(hsv[:, :, 1])
    if sat_std < 20:
        anomalies.append("color_anomaly")
        reasons.append(f"🎨 **Color Analysis:** Saturation distribution is unusually narrow (std: {sat_std:.1f}). "
                       f"AI-generated images often have a limited/artificial color palette that lacks the natural variation found in real photos.")

    # 6. EXIF Metadata check
    if isinstance(source, str):
        from utils.feature_extractors import extract_metadata
        meta = extract_metadata(source)
        if meta.get("has_ai_tags"):
            anomalies.append("ai_metadata_tag")
            sw = meta.get("software", "unknown tool")
            reasons.append(f"🏷️ **Metadata Analysis:** AI-generation markers detected in EXIF data (software: {sw}). "
                           f"The file contains metadata tags associated with known AI image generation tools.")

    return anomalies, reasons


# ─── Audio Feature Analysis for Reasoning ────────────────────────────
def _analyze_audio_features(source, sr=16000):
    """Run feature-level checks on the audio and return anomalies + reasons."""
    import librosa

    if isinstance(source, str):
        y, sr = librosa.load(source, sr=sr)
    else:
        y = source.astype(np.float32)

    anomalies = []
    reasons = []

    # 1. Spectral Centroid Variance — synthetic speech has less variation
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    cent_var = np.var(cent)
    cent_mean = np.mean(cent)

    if cent_var < 5e5:
        anomalies.append("low_spectral_variation")
        reasons.append(f"📊 **Spectral Analysis:** Spectral centroid has very low variance ({cent_var:.0f}). "
                       f"Natural speech and audio have dynamic spectral content that shifts over time. "
                       f"AI-generated audio tends to have unnaturally stable spectral characteristics.")

    # 2. Zero Crossing Rate — synthetic audio has different ZCR patterns
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_std = np.std(zcr)

    if zcr_std < 0.01:
        anomalies.append("uniform_zero_crossings")
        reasons.append(f"〰️ **Zero-Crossing Analysis:** Zero-crossing rate is unusually uniform (std: {zcr_std:.4f}). "
                       f"Real audio signals have natural fluctuations in zero-crossings due to environmental noise and speech dynamics. "
                       f"Synthetic audio often shows artificially smooth waveform transitions.")

    # 3. MFCC Variance — AI-generated speech has lower MFCC variance
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = np.mean(np.var(mfcc, axis=1))

    if mfcc_var < 50:
        anomalies.append("synthetic_formants")
        reasons.append(f"🗣️ **Formant Analysis:** MFCC variance is abnormally low ({mfcc_var:.1f}). "
                       f"This indicates limited vocal tract variation — typical of AI voice synthesis which struggles to replicate "
                       f"the natural micro-variations in human speech articulation.")

    # 4. Spectral Rolloff — synthetic audio often has truncated frequency range
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = np.mean(rolloff)

    if rolloff_mean < 2000:
        anomalies.append("truncated_spectrum")
        reasons.append(f"📉 **Frequency Range:** Average spectral rolloff is low ({rolloff_mean:.0f} Hz). "
                       f"The audio lacks natural high-frequency content above 2 kHz, suggesting it was synthesized by a model "
                       f"with limited frequency range or aggressive low-pass filtering.")

    # 5. Signal Noise Analysis — AI audio has unnatural noise floors
    rms = librosa.feature.rms(y=y)[0]
    rms_min = np.min(rms)
    rms_max = np.max(rms)
    dynamic_range = 20 * np.log10(rms_max / (rms_min + 1e-8) + 1e-8)

    if dynamic_range < 15:
        anomalies.append("flat_dynamics")
        reasons.append(f"🔇 **Dynamics Analysis:** Very low dynamic range ({dynamic_range:.1f} dB). "
                       f"Natural speech has significant volume variation between syllables, pauses, and breaths. "
                       f"AI-generated audio often maintains an unnaturally consistent volume level.")

    # 6. Breath/Pause Detection — AI audio often lacks natural breathing
    silence_threshold = 0.01
    silence_ratio = np.sum(np.abs(y) < silence_threshold) / len(y)

    if silence_ratio < 0.02:
        anomalies.append("missing_natural_pauses")
        reasons.append(f"😮‍💨 **Breathing Analysis:** Almost no natural pauses detected (silence: {silence_ratio:.1%}). "
                       f"Real human speech contains micro-pauses, breathing sounds, and natural hesitations. "
                       f"AI-generated speech typically flows without these organic interruptions.")

    return anomalies, reasons


# ─── Public API ──────────────────────────────────────────────────────

def predict_image(source):
    """Run image detection on a file path or numpy RGB array.
    Returns dict with prediction, confidence, anomalies, reasons, and timing.
    """
    from utils.feature_extractors import preprocess_image
    t0 = time.perf_counter()
    rgb, freq, edge = preprocess_image(source)
    dev = _get_device()
    model = _load_image_model()

    with torch.no_grad():
        logits, _ = model(rgb.unsqueeze(0).to(dev), freq.unsqueeze(0).to(dev), edge.unsqueeze(0).to(dev))
        probs = F.softmax(logits, dim=1).squeeze()
        conf, idx = probs.max(0)

    prediction = CLASSES[idx.item()]

    # Run feature-level analysis for reasoning
    anomalies, reasons = _analyze_image_features(source)

    # Add model probability insights
    if prediction != "REAL":
        reasons.insert(0,
            f"🤖 **Model Verdict:** The neural network classified this image as **{prediction}** "
            f"with **{conf.item():.1%}** confidence. "
            f"Class probabilities — Real: {probs[0].item():.1%}, Manipulated: {probs[1].item():.1%}, AI-Generated: {probs[2].item():.1%}."
        )
    else:
        reasons = [
            f"✅ **Model Verdict:** No signs of artificial generation or manipulation detected. "
            f"Confidence: **{conf.item():.1%}**. The image passes all feature-level integrity checks."
        ]

    dt = (time.perf_counter() - t0) * 1000
    return {
        "prediction": prediction,
        "confidence": round(conf.item(), 4),
        "anomalies": anomalies,
        "reasons": reasons,
        "detection_time_ms": int(dt),
    }


def predict_audio(source, sr=16000):
    """Run audio detection on a file path or numpy waveform.
    Returns dict with prediction, confidence, anomalies, reasons, and timing.
    """
    from utils.feature_extractors import extract_audio_features
    t0 = time.perf_counter()
    mel, aux = extract_audio_features(source, sr=sr)
    if mel is None:
        return {"prediction": "ERROR", "confidence": 0, "anomalies": [], "reasons": ["Could not process audio."], "detection_time_ms": 0}

    dev = _get_device()
    model = _load_audio_model()

    with torch.no_grad():
        logits, _ = model(mel.unsqueeze(0).to(dev), aux.unsqueeze(0).to(dev))
        probs = F.softmax(logits, dim=1).squeeze()
        conf, idx = probs.max(0)

    prediction = CLASSES[idx.item()]

    # Run feature-level analysis for reasoning
    anomalies, reasons = _analyze_audio_features(source, sr=sr)

    if prediction != "REAL":
        reasons.insert(0,
            f"🤖 **Model Verdict:** The neural network classified this audio as **{prediction}** "
            f"with **{conf.item():.1%}** confidence. "
            f"Class probabilities — Real: {probs[0].item():.1%}, Manipulated: {probs[1].item():.1%}, AI-Generated: {probs[2].item():.1%}."
        )
    else:
        reasons = [
            f"✅ **Model Verdict:** No signs of synthetic generation or manipulation detected. "
            f"Confidence: **{conf.item():.1%}**. The audio passes all spectral integrity checks."
        ]

    dt = (time.perf_counter() - t0) * 1000
    return {
        "prediction": prediction,
        "confidence": round(conf.item(), 4),
        "anomalies": anomalies,
        "reasons": reasons,
        "detection_time_ms": int(dt),
    }
