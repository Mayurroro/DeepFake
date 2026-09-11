"""
Deepfake Live Detector — Streamlit Dashboard
Real-time webcam + microphone deepfake detection with file upload support.
Launch:  streamlit run frontend/streamlit_app.py --server.port 8501
"""
import sys, os, time, io
from pathlib import Path
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from realtime.live_inference import predict_image, predict_audio

# ─── Page Config ────────────────────────────────────────────────────
st.set_page_config(page_title="Deepfake Live Detector", page_icon="🛡️", layout="wide")

# ─── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d2b 100%); }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d2b 100%); }
[data-testid="stSidebar"] { background: rgba(15,15,40,0.95); border-right: 1px solid rgba(100,100,255,0.15); }
h1, h2, h3, p, span, label, .stMarkdown { color: #e8e8ff !important; }
.hero-title { font-size: 2.8rem; font-weight: 900; background: linear-gradient(135deg, #00d4ff, #7b2fff, #ff2d95);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0.2rem; }
.hero-sub { text-align: center; color: #888aaf !important; font-size: 1.1rem; margin-bottom: 2rem; }
.glass-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px; padding: 1.5rem; backdrop-filter: blur(12px); margin-bottom: 1rem; }
.result-real { background: linear-gradient(135deg, rgba(0,220,130,0.15), rgba(0,180,100,0.05));
  border: 1px solid rgba(0,220,130,0.3); border-radius: 12px; padding: 1rem; }
.result-fake { background: linear-gradient(135deg, rgba(255,45,80,0.15), rgba(255,0,50,0.05));
  border: 1px solid rgba(255,45,80,0.3); border-radius: 12px; padding: 1rem; }
.result-manip { background: linear-gradient(135deg, rgba(255,180,0,0.15), rgba(255,140,0,0.05));
  border: 1px solid rgba(255,180,0,0.3); border-radius: 12px; padding: 1rem; }
.metric-box { text-align: center; padding: 0.8rem; border-radius: 12px;
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); }
.anomaly-tag { display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 20px;
  background: rgba(123,47,255,0.2); border: 1px solid rgba(123,47,255,0.4);
  color: #c4a8ff; font-size: 0.8rem; }
.reason-box { background: rgba(20,20,50,0.6); border-left: 3px solid rgba(123,47,255,0.6);
  padding: 0.8rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0; font-size: 0.92rem; line-height: 1.6; }
.reason-box strong { color: #c4a8ff !important; }
.stButton > button { background: linear-gradient(135deg, #7b2fff, #00d4ff) !important;
  color: white !important; border: none !important; border-radius: 12px !important;
  padding: 0.6rem 2rem !important; font-weight: 700 !important; font-size: 1rem !important;
  transition: all 0.3s ease !important; }
.stButton > button:hover { transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(123,47,255,0.35) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🛡️ Deepfake Live Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-time AI-powered detection for images & audio — Camera • Microphone • Upload</div>', unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.markdown("**Supported Formats**")
    st.markdown("🖼️ JPG, PNG, WebP, HEIC")
    st.markdown("🔊 MP3, WAV, M4A, FLAC")
    st.markdown("---")
    st.markdown("**Performance Targets**")
    st.markdown("• Image: < 150 ms")
    st.markdown("• Audio: < 100 ms")
    

# ─── History ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

def _result_card(res, source_label):
    pred = res["prediction"]
    conf = res["confidence"]
    css = "result-real" if pred == "REAL" else ("result-fake" if pred == "AI_GENERATED" else "result-manip")
    icon = "✅" if pred == "REAL" else ("🚨" if pred == "AI_GENERATED" else "⚠️")
    st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.markdown(f"### {icon} {pred}")
        st.caption(f"Source: {source_label}")
    with c2:
        st.metric("Confidence", f"{conf:.1%}")
        st.progress(conf)
    with c3:
        st.metric("Latency", f"{res['detection_time_ms']} ms")
    if res.get("anomalies"):
        tags = "".join(f'<span class="anomaly-tag">{a}</span>' for a in res["anomalies"])
        st.markdown(f"**Anomalies:** {tags}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Detailed Reasoning Section ──
    reasons = res.get("reasons", [])
    if reasons:
        label = "🔍 Why is this flagged?" if pred != "REAL" else "🔍 Analysis Details"
        with st.expander(label, expanded=(pred != "REAL")):
            for reason in reasons:
                st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

    history_entry = {k: v for k, v in res.items() if k != "reasons"}
    history_entry["source"] = source_label
    history_entry["time"] = time.strftime("%H:%M:%S")
    st.session_state.history.insert(0, history_entry)


# ─── Tabs ───────────────────────────────────────────────────────────
tab_cam, tab_mic, tab_upload, tab_hist = st.tabs(["📷 Camera", "🎙️ Microphone", "📁 Upload", "📊 History"])

# ── Camera Tab ──────────────────────────────────────────────────────
with tab_cam:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📷 Live Camera Detection")
    st.markdown("Capture a photo from your webcam for instant AI deepfake analysis.")

    cam_img = st.camera_input("Point your camera and press capture")

    if cam_img is not None:
        import cv2
        from PIL import Image as PILImage
        pil = PILImage.open(cam_img)
        arr = np.array(pil)

        with st.spinner("⚡ Analyzing captured frame..."):
            res = predict_image(arr)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.image(arr, caption="Captured Frame", use_container_width=True)
        with col_b:
            _result_card(res, "camera")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Microphone Tab ──────────────────────────────────────────────────
with tab_mic:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Microphone Detection")
    st.markdown("Record a short audio clip for deepfake voice analysis.")

    audio_bytes = st.audio_input("Press the microphone button to record")

    if audio_bytes is not None:
        # Save to temp file for librosa processing
        tmp_audio = ROOT / "tmp" / "mic_recording.wav"
        tmp_audio.parent.mkdir(exist_ok=True)
        with open(tmp_audio, "wb") as f:
            f.write(audio_bytes.getvalue())

        st.audio(audio_bytes, format="audio/wav")

        with st.spinner("⚡ Analyzing audio..."):
            res = predict_audio(str(tmp_audio))

        _result_card(res, "microphone")
        tmp_audio.unlink(missing_ok=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Upload Tab ──────────────────────────────────────────────────────
with tab_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📁 File Upload")
    st.markdown("Drag & drop images or audio files for detection.")

    uploaded = st.file_uploader(
        "Choose files", type=["jpg", "jpeg", "png", "webp", "mp3", "wav", "m4a", "flac"],
        accept_multiple_files=True,
    )

    if uploaded:
        for uf in uploaded:
            ext = uf.name.rsplit(".", 1)[-1].lower()
            tmp_path = ROOT / "tmp" / uf.name
            tmp_path.parent.mkdir(exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(uf.getvalue())

            with st.spinner(f"⚡ Analyzing {uf.name}..."):
                if ext in {"jpg", "jpeg", "png", "webp", "heic", "bmp"}:
                    st.image(uf, width=300)
                    res = predict_image(str(tmp_path))
                elif ext in {"mp3", "wav", "m4a", "flac"}:
                    st.audio(uf)
                    res = predict_audio(str(tmp_path))
                else:
                    st.warning(f"Unsupported: {uf.name}")
                    continue

            _result_card(res, f"upload: {uf.name}")
            tmp_path.unlink(missing_ok=True)
            st.markdown("---")
    st.markdown("</div>", unsafe_allow_html=True)

# ── History Tab ─────────────────────────────────────────────────────
with tab_hist:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Detection History")
    if st.session_state.history:
        import pandas as pd
        df = pd.DataFrame(st.session_state.history)
        display_cols = [c for c in ["time", "source", "prediction", "confidence", "detection_time_ms", "anomalies"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # Stats row
        c1, c2, c3 = st.columns(3)
        total = len(df)
        real_pct = len(df[df["prediction"] == "REAL"]) / total * 100 if total else 0
        avg_ms = df["detection_time_ms"].mean() if total else 0
        with c1:
            st.markdown(f'<div class="metric-box"><h3>{total}</h3><p>Total Scans</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><h3>{real_pct:.0f}%</h3><p>Real Content</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><h3>{avg_ms:.0f} ms</h3><p>Avg Latency</p></div>', unsafe_allow_html=True)
    else:
        st.info("No detections yet. Use the Camera, Microphone, or Upload tabs to start.")
    st.markdown("</div>", unsafe_allow_html=True)
