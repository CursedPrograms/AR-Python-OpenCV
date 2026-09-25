"""MiDaS v2.1 monocular depth (ONNX), used by ar_depth.py.

MiDaS predicts *relative inverse* depth: bigger = closer, with an unknown scale per
frame. to_metres() turns it into metres using one known distance (the ArUco marker).
The model is downloaded to models/ on first use.
"""
import os
import sys

import cv2
import numpy as np

import ar_common

MODEL_DIR = os.path.join(ar_common.REPO_DIR, "models")
MODEL_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/{}"
MODELS = {
    "small": ("model-small.onnx", 256),     # ~63 MB, ~0.1 s per frame on a CPU
    "large": ("model-f6b98070.onnx", 384),  # ~400 MB, ~2 s per frame on a CPU
}
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def download_model(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(path):
        return path
    import requests

    os.makedirs(MODEL_DIR, exist_ok=True)
    part = path + ".part"
    print(f"Downloading {file_name} (first run only)...")
    try:
        with requests.get(MODEL_URL.format(file_name), stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / total:6.1%}", end="", flush=True)
        print()
        os.replace(part, path)
    except Exception as e:
        if os.path.exists(part):
            os.remove(part)
        sys.exit(f"Could not download the depth model: {e}")
    return path


class DepthEstimator:
    def __init__(self, model="small"):
        file_name, self.input_size = MODELS[model]
        path = download_model(file_name)
        try:
            import onnxruntime
            self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.net = None
        except ImportError:
            self.session = None
            self.net = cv2.dnn.readNet(path)

    def raw(self, frame_bgr):
        """Relative inverse depth at the frame's size (float32, bigger = closer)."""
        h, w = frame_bgr.shape[:2]
        # MiDaS input: RGB, (x/255 - mean) / std. blobFromImage does the mean; divide by std after.
        blob = cv2.dnn.blobFromImage(frame_bgr, 1 / 255., (self.input_size, self.input_size),
                                     (123.675, 116.28, 103.53), True, False)
        blob /= IMAGENET_STD
        if self.session is not None:
            out = self.session.run(None, {self.input_name: blob})[0][0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()[0]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def metric_scale(raw, known_mask, known_depth_m):
    """Scale that puts the pixels in known_mask (e.g. the marker) at known_depth_m,
    or None. MiDaS output is roughly proportional to 1 / depth, so
    depth = scale / raw; this holds well near the reference and loosely far from it."""
    ref = np.median(raw[known_mask]) if known_mask.any() else 0
    return known_depth_m * ref if ref > 0 else None


def to_metres(raw, scale):
    """Metric depth map (metres) from MiDaS output and a metric_scale()."""
    return scale / np.maximum(raw, 1e-6)


def colorize(raw):
    d = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
