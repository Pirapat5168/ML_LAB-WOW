"""Download a built-in Ultralytics detection dataset into ./dataset (YOLO format).

    python download_dataset.py --name african-wildlife     # 4 classes, small (recommended)
    python download_dataset.py --name brain-tumor          # 2 classes
    python download_dataset.py --name signature            # 1 class

Result: dataset/{train,val,test}/{images,labels} + dataset/data.yaml
(the test split is skipped automatically if the dataset has none).
"""
import argparse
import shutil
from pathlib import Path

from ultralytics.data.utils import check_det_dataset

from common import DATASET

CHOICES = ["african-wildlife", "brain-tumor", "signature"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="african-wildlife", choices=CHOICES)
    a = ap.parse_args()

    info = check_det_dataset(f"{a.name}.yaml")      # downloads + unzips if needed
    names = info["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else names

    lines = []
    for split in ("train", "val", "test"):
        entry = info.get(split)
        if not entry:
            print(f"[INFO] no '{split}' split in {a.name}")
            continue
        src_img = Path(entry[0] if isinstance(entry, list) else entry)
        src_lbl = Path(str(src_img).replace("images", "labels"))
        dst = DATASET / split
        for s, d in ((src_img, dst / "images"), (src_lbl, dst / "labels")):
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
        n = sum(1 for _ in (dst / "images").iterdir())
        print(f"{split}: {n} images")
        lines.append(f"{split}: {split}/images")

    lines += ["", f"nc: {len(names)}", "", "names:"]
    lines += [f"  {int(k)}: {v}" for k, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    (DATASET / "data.yaml").write_text("\n".join(lines) + "\n")
    print(f"Wrote {DATASET / 'data.yaml'}  ({len(names)} classes: {', '.join(names.values())})")


if __name__ == "__main__":
    main()
