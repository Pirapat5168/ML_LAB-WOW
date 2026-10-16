"""Dataset helper (Lab step: select / prepare / split a YOLO-format dataset).

1) Check an existing dataset (e.g. BDD100K from Kaggle, already split):
       python prepare_dataset.py --check

2) Split a flat YOLO dataset (images/ + labels/) into train/val/test
   and write dataset/data.yaml  (use this for your own / Lab 7 dataset):
       python prepare_dataset.py --images raw/images --labels raw/labels \
              --names person car truck --ratios 0.7 0.2 0.1
   (--names can also be a classes.txt file path, one class per line)
"""
import argparse
import random
import shutil
from pathlib import Path

from common import DATASET, IMG_EXT


def check():
    print(f"Dataset folder: {DATASET}")
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = DATASET / split / "images", DATASET / split / "labels"
        n_img = len([p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXT]) if img_dir.is_dir() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.is_dir() else 0
        print(f"  {split:5s}: {n_img:7d} images | {n_lbl:7d} label files")


def load_names(values):
    if len(values) == 1 and Path(values[0]).is_file():
        return [l.strip() for l in Path(values[0]).read_text().splitlines() if l.strip()]
    return values


def split_dataset(images, labels, names, ratios, seed):
    images, labels = Path(images), Path(labels)
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in IMG_EXT)
    files = [p for p in files if (labels / f"{p.stem}.txt").exists()]
    if not files:
        raise SystemExit("No image/label pairs found.")
    random.Random(seed).shuffle(files)

    n = len(files)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    parts = {"train": files[:n_tr], "val": files[n_tr:n_tr + n_va], "test": files[n_tr + n_va:]}

    for split, items in parts.items():
        (DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET / split / "labels").mkdir(parents=True, exist_ok=True)
        for p in items:
            shutil.copy2(p, DATASET / split / "images" / p.name)
            shutil.copy2(labels / f"{p.stem}.txt", DATASET / split / "labels" / f"{p.stem}.txt")
        print(f"{split}: {len(items)} images")

    lines = ["train: train/images", "val: val/images", "test: test/images", "",
             f"nc: {len(names)}", "", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (DATASET / "data.yaml").write_text("\n".join(lines) + "\n")
    print(f"Wrote {DATASET / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="count images/labels per split")
    ap.add_argument("--images")
    ap.add_argument("--labels")
    ap.add_argument("--names", nargs="+", help="class names, or a classes.txt file")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.check or not (a.images and a.labels):
        check()
        return
    if not a.names:
        raise SystemExit("--names is required when splitting")
    split_dataset(a.images, a.labels, load_names(a.names), a.ratios, a.seed)
    check()


if __name__ == "__main__":
    main()
