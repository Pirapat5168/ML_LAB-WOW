
from pathlib import Path

import cv2
import numpy as np

from preprocessing import preprocess_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _read_image(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII paths on Windows
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_dataset(data_dir, img_size=96, max_per_class=None, seed=42):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    rng = np.random.default_rng(seed)
    X, y = [], []
    skipped = 0

    for label, cls in enumerate(classes):
        files = [f for f in (data_dir / cls).iterdir() if f.suffix.lower() in IMG_EXTS]
        files.sort()
        if max_per_class:
            rng.shuffle(files)
        loaded = 0
        for f in files:
            if max_per_class and loaded >= max_per_class:
                break
            img = _read_image(f)
            if img is None:
                skipped += 1
                continue
            X.append(preprocess_image(img, img_size))
            y.append(label)
            loaded += 1
            if loaded % 1000 == 0:
                print(f"  {cls}: {loaded} images loaded...")
        print(f"[{cls}] loaded {loaded} images")

    print(f"Skipped corrupted/unreadable files: {skipped}")
    return np.asarray(X, dtype=np.uint8), np.asarray(y, dtype=np.int64), classes
