"""Evaluate the Fast R-CNN detector: mAP@0.5 (real/fake), image-level accuracy, precision/recall/F1.

    python evaluate.py --limit 200 --split test
"""
from pathlib import Path
import sys
import json
import argparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import evaluate_detector   # noqa: E402
from train import load_model, detect, OUT   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=200, help="max images (0 = all)")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--regions", type=int, default=150, help="region proposals per image")
    a = p.parse_args()

    model = load_model()
    result = evaluate_detector(lambda img, n, thr: detect(model, img, n, thr), a.split, a.limit, a.regions)
    result["model"] = "Fast R-CNN"
    print(json.dumps(result, indent=2))
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2))
    print("Saved:", OUT / "evaluation.json")


if __name__ == "__main__":
    main()
