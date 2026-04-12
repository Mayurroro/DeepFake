"""Feature extraction for image and audio deepfake detection."""
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from PIL.ExifTags import TAGS

# ── Image ──────────────────────────────────────────────────────────

IMG_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(source, size=(512, 512)):
    """Accept file-path *or* BGR numpy array. Returns (rgb_tensor, freq_tensor, edge_tensor)."""
    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"Cannot read image: {source}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(source, np.ndarray):
        img = source if source.shape[2] == 3 else cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError("source must be a filepath or numpy array")

    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # FFT magnitude spectrum
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = 20 * np.log(np.abs(f) + 1e-8)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    # Canny edges
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 100, 200) / 255.0

    rgb_t = IMG_TRANSFORM(img)
    freq_t = torch.tensor(mag, dtype=torch.float32).unsqueeze(0)
    edge_t = torch.tensor(edges, dtype=torch.float32).unsqueeze(0)
    return rgb_t, freq_t, edge_t


def extract_metadata(path):
    """Scan EXIF for AI-generation markers."""
    meta = {"has_ai_tags": False, "software": ""}
    try:
        exif = Image.open(path).getexif()
        for tid, val in exif.items():
            tag = TAGS.get(tid, tid)
            if isinstance(val, bytes):
                try: val = val.decode()
                except Exception: continue
            if isinstance(val, str):
                lo = val.lower()
                if any(k in lo for k in ("midjourney", "dall-e", "stable diffusion", "ai generated", "novelai")):
                    meta["has_ai_tags"] = True
                if tag == "Software":
                    meta["software"] = val
    except Exception:
        pass
    return meta


# ── Audio ──────────────────────────────────────────────────────────

def extract_audio_features(source, sr=16000, max_time_steps=400):
    """Accept file-path *or* numpy waveform. Returns (mel_tensor[1,128,T], aux_tensor[30])."""
    import librosa

    if isinstance(source, str):
        y, sr = librosa.load(source, sr=sr)
    else:
        y = source.astype(np.float32)

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
    ])  # length = 2+2+13+13 = 30

    mel_t = torch.tensor(S, dtype=torch.float32).unsqueeze(0)
    aux_t = torch.tensor(aux, dtype=torch.float32)
    return mel_t, aux_t
