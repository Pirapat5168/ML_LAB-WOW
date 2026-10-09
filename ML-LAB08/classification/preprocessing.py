
import cv2
import numpy as np


def preprocess_image(img_bgr, img_size):
    """Resize to (img_size, img_size) and convert BGR (OpenCV) to RGB."""
    img = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_channel_stats(X, chunk=500):
    
    s = np.zeros(3, dtype=np.float64)
    sq = np.zeros(3, dtype=np.float64)
    n = 0
    for i in range(0, len(X), chunk):
        c = X[i:i + chunk].astype(np.float64)
        s += c.sum(axis=(0, 1, 2))
        sq += (c ** 2).sum(axis=(0, 1, 2))
        n += c.shape[0] * c.shape[1] * c.shape[2]
    mean = s / n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 1e-12))
    return mean.astype("float32"), std.astype("float32")
