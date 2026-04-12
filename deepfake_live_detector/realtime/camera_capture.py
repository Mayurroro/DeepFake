"""OpenCV-based webcam capture with frame buffering for real-time detection."""
import cv2
import time
import threading
import numpy as np


class CameraCapture:
    """Thread-safe camera capture with frame buffering.
    
    Usage:
        cam = CameraCapture(device=0)
        cam.start()
        frame = cam.read()   # latest BGR frame
        cam.stop()
    """

    def __init__(self, device=0, width=640, height=480):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            time.sleep(0.01)  # ~100 FPS cap

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def capture_rgb(self):
        """Grab a single frame as RGB numpy array (H,W,3)."""
        f = self.read()
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f is not None else None

    def stop(self):
        self._running = False
        self.cap.release()

    @property
    def is_opened(self):
        return self.cap.isOpened()
