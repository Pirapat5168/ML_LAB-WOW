"""Shared paths and experiment settings for Lab 10 (YOLO object detection)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
DATA_YAML = DATASET / "data.yaml"
RUNS = ROOT / "runs"
SUMMARY = RUNS / "summary"

# Training configurations to compare (all fine-tune yolov8n.pt).
#   Group A (epochs):  ep10 vs ep30 vs ep50      (imgsz=480, batch=8)
#   Group B (configs): ep30 vs img640_b8 vs img480_b16 (30 epochs)
EXPERIMENTS = {
    "ep10":       dict(epochs=10, imgsz=480, batch=8),
    "ep30":       dict(epochs=30, imgsz=480, batch=8),
    "ep50":       dict(epochs=50, imgsz=480, batch=8),
    "img640_b8":  dict(epochs=30, imgsz=640, batch=8),
    "img480_b16": dict(epochs=30, imgsz=480, batch=16),
}
# Which trained model the inference scripts use by default (first one found)
DEFAULT_ORDER = ["ep50", "ep30", "ep10", "img640_b8", "img480_b16"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def get_device():
    """GPU 0 if CUDA is available (WSL2 + NVIDIA driver), otherwise CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    print("[WARN] CUDA not available -> using CPU (very slow for training).")
    return "cpu"


def exp_dir(name):
    return RUNS / f"exp_{name}"


def epochs_done(name):
    """Number of epochs already written to results.csv (0 if none)."""
    f = exp_dir(name) / "results.csv"
    if not f.exists():
        return 0
    with open(f, newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def existing_experiments():
    return [n for n in EXPERIMENTS
            if (exp_dir(n) / "weights" / "best.pt").exists()]


def resolve_weights(path=None):
    """Explicit --weights path, else the first trained experiment found."""
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Weights not found: {p}")
        return p
    for n in DEFAULT_ORDER:
        p = exp_dir(n) / "weights" / "best.pt"
        if p.exists():
            return p
    raise FileNotFoundError("No trained model found. Run train.py first "
                            "or pass --weights path/to/best.pt")


def pick_eval_split(preferred="test"):
    """Use the test split if it has label files, otherwise fall back to val."""
    labels = DATASET / preferred / "labels"
    if labels.is_dir() and any(labels.glob("*.txt")):
        return preferred
    print(f"[WARN] '{preferred}' split has no label files -> evaluating on 'val'.")
    return "val"


def pick_image_dir(preferred="test"):
    d = DATASET / preferred / "images"
    if d.is_dir() and any(p.suffix.lower() in IMG_EXT for p in d.iterdir()):
        return d
    d = DATASET / "val" / "images"
    if d.is_dir():
        print("[WARN] no test images found -> using val images.")
        return d
    raise FileNotFoundError(f"No images found under {DATASET}")
