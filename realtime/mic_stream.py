"""Microphone streaming with rolling buffer for real-time audio deepfake detection."""
import numpy as np
import threading
import time

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


class MicStream:
    """Records audio from the default microphone in a rolling buffer.
    
    Usage:
        mic = MicStream()
        mic.start()
        chunk = mic.get_chunk()  # latest 3-second numpy array
        mic.stop()
    """

    def __init__(self, sr=SAMPLE_RATE, chunk_seconds=CHUNK_SECONDS):
        self.sr = sr
        self.chunk_samples = sr * chunk_seconds
        self._buffer = np.zeros(self.chunk_samples, dtype=np.float32)
        self._lock = threading.Lock()
        self._running = False
        self._stream = None

    def start(self):
        try:
            import sounddevice as sd
            self._running = True
            self._stream = sd.InputStream(
                samplerate=self.sr, channels=1, dtype="float32",
                blocksize=1024, callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[MicStream] Could not open microphone: {e}")
            self._running = False
        return self

    def _callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0]
        with self._lock:
            self._buffer = np.roll(self._buffer, -len(chunk))
            self._buffer[-len(chunk):] = chunk

    def get_chunk(self):
        """Return the latest 3-second audio as float32 numpy array."""
        with self._lock:
            return self._buffer.copy()

    def get_rms(self):
        """Return current RMS level (voice activity indicator)."""
        with self._lock:
            return float(np.sqrt(np.mean(self._buffer[-1600:] ** 2)))

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
