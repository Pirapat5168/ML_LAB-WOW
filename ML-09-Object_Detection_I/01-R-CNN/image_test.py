"""Run the R-CNN detector on one image and save the result (green = real, red = fake, orange = ground truth).

    python image_test.py                       # random test image
    python image_test.py --image path/to/face.jpg
"""
from pathlib import Path
import sys
import argparse
import random
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import list_images, read_labels, draw, NAMES   # noqa: E402
from train import load_model, detect, OUT   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--regions", type=int, default=150)
    a = p.parse_args()

    path = Path(a.image) if a.image else random.choice(list_images(a.split))
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(str(path))
    h, w = img.shape[:2]
    gt = read_labels(path, w, h)
    gt = gt if len(gt[0]) else None

    dets = detect(load_model(), img, a.regions, a.threshold)
    save = OUT / "image_result.jpg"
    cv2.imwrite(str(save), draw(img, dets, gt))
    print("Input:", path)
    if gt is not None:
        print("Ground truth:", [NAMES[int(c)] for c in gt[1]])
    for x1, y1, x2, y2, c, s in dets[:5]:
        print(f"  {NAMES[int(c)]} {s:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    if not dets:
        print("  no detection above threshold", a.threshold)
    print("Saved:", save)


if __name__ == "__main__":
    main()
