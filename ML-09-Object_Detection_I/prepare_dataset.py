"""Convert the Lab 7 real-vs-fake faces dataset into a YOLO-format detection dataset.

The Lab 7 data has class labels only (no bounding boxes), so a face box is
auto-generated for every image with OpenCV's Haar cascade (pseudo ground truth).
If no face is found, a centered box covering most of the image is used
(the images are usually face-centric).  YOLO class ids: 0 = real, 1 = fake.

Usage (from ML-09-Object_Detection_I/):
    python prepare_dataset.py --data-dir "C:/Users/NITRO V15/Documents/ML/ML-07-CNN/ML-07-CNN/DATASET"
    python prepare_dataset.py --data-dir ... --max-per-class 1500      # quick subset
"""
from pathlib import Path
import argparse
import json
import random
import shutil
import cv2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dataset"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
NAMES = ["real", "fake"]


def class_of(path, base):
    parts = [p.lower() for p in path.relative_to(base).parts[:-1]]
    is_fake = any("fake" in p for p in parts)
    is_real = any("real" in p for p in parts)
    if is_fake == is_real:
        return None
    return 1 if is_fake else 0


def make_detector():
    base = cv2.data.haarcascades
    return [cv2.CascadeClassifier(base + n)
            for n in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")]


def find_face(img, cascades, margin):
    h, w = img.shape[:2]
    scale = 320.0 / max(h, w) if max(h, w) > 320 else 1.0
    gray = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    for casc in cascades:
        faces = casc.detectMultiScale(gray, 1.1, 4, minSize=(int(24), int(24)))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = x / scale, y / scale, fw / scale, fh / scale
            x1, y1 = max(0, x - margin * fw), max(0, y - margin * fh)
            x2, y2 = min(w, x + fw * (1 + margin)), min(h, y + fh * (1 + margin))
            return (x1, y1, x2, y2), True
    m = 0.05
    return (m * w, m * h, (1 - m) * w, (1 - m) * h), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Lab 7 DATASET folder (contains real/fake class folders)")
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = use all images")
    ap.add_argument("--margin", type=float, default=0.10, help="extra margin around Haar face box")
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(args.data_dir)
    if not base.is_dir():
        raise FileNotFoundError(base)
    per_class = {0: [], 1: []}
    skipped = 0
    for p in sorted(base.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            c = class_of(p, base)
            if c is None:
                skipped += 1
            else:
                per_class[c].append(p)
    print({NAMES[c]: len(v) for c, v in per_class.items()}, "skipped (class unclear):", skipped)
    if not per_class[0] or not per_class[1]:
        raise RuntimeError("Need folders whose names contain 'real' and 'fake' under --data-dir")

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for c, files in per_class.items():               # stratified split
        rng.shuffle(files)
        if args.max_per_class > 0:
            files = files[:args.max_per_class]
        n = len(files); a = int(n * args.split[0]); b = a + int(n * args.split[1])
        splits["train"] += [(f, c) for f in files[:a]]
        splits["val"] += [(f, c) for f in files[a:b]]
        splits["test"] += [(f, c) for f in files[b:]]

    if OUT.exists():
        shutil.rmtree(OUT)
    cascades = make_detector()
    report = {}
    for name, items in splits.items():
        (OUT / name / "images").mkdir(parents=True)
        (OUT / name / "labels").mkdir(parents=True)
        found = 0
        for i, (src, c) in enumerate(items):
            img = cv2.imread(str(src))
            if img is None:
                continue
            h, w = img.shape[:2]
            (x1, y1, x2, y2), ok = find_face(img, cascades, args.margin)
            found += ok
            stem = f"{NAMES[c]}_{i:05d}"
            cv2.imwrite(str(OUT / name / "images" / f"{stem}.jpg"), img)
            xc, yc, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
            (OUT / name / "labels" / f"{stem}.txt").write_text(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            if (i + 1) % 500 == 0:
                print(f"  {name}: {i + 1}/{len(items)}")
        report[name] = {"images": len(items), "haar_face_found": found,
                        "fallback_center_box": len(items) - found}
        print(name, report[name])

    (OUT / "data.yaml").write_text(
        "path: dataset\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "nc: 2\nnames: [real, fake]\n")
    (OUT / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print("Dataset ready:", OUT)


if __name__ == "__main__":
    main()
