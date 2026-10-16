"""Evaluate trained experiments: Precision, Recall, mAP50, mAP50-95.

    python evaluate.py                    # every experiment that has best.pt
    python evaluate.py --exp ep10 ep50
    python evaluate.py --split val

Outputs: runs/summary/evaluation_summary.csv  (+ per_class_<exp>.csv)
         runs/evaluation/<exp>/  (confusion matrix, PR / F1 curves, ...)
"""
import argparse

import pandas as pd
from ultralytics import YOLO

from common import (DATA_YAML, EXPERIMENTS, RUNS, SUMMARY, exp_dir,
                    existing_experiments, get_device, pick_eval_split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", choices=list(EXPERIMENTS))
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    names = args.exp or existing_experiments()
    if not names:
        raise SystemExit("No trained experiments found. Run train.py first.")

    split = pick_eval_split(args.split) if args.split == "test" else "val"
    device = get_device()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    rows = []

    for name in names:
        cfg = EXPERIMENTS[name]
        model = YOLO(str(exp_dir(name) / "weights" / "best.pt"))
        m = model.val(
            data=str(DATA_YAML), split=split,
            imgsz=cfg["imgsz"], batch=cfg["batch"],
            device=device, workers=args.workers, plots=True,
            project=str(RUNS / "evaluation"), name=name, exist_ok=True,
        )
        rows.append({
            "experiment": name, "epochs": cfg["epochs"],
            "imgsz": cfg["imgsz"], "batch": cfg["batch"], "split": split,
            "precision": m.box.mp, "recall": m.box.mr,
            "mAP50": m.box.map50, "mAP50-95": m.box.map,
            "inference_ms": m.speed["inference"],
        })

        per_class = []
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            per_class.append({"class": m.names[int(c)], "precision": p,
                              "recall": r, "AP50": ap50, "AP50-95": ap})
        pd.DataFrame(per_class).round(4).to_csv(SUMMARY / f"per_class_{name}.csv", index=False)

    df = pd.DataFrame(rows).round(4)
    df.to_csv(SUMMARY / "evaluation_summary.csv", index=False)
    print("\n===== Evaluation Results =====")
    print(df.to_string(index=False))
    print(f"\nSaved: {SUMMARY / 'evaluation_summary.csv'}")


if __name__ == "__main__":
    main()
