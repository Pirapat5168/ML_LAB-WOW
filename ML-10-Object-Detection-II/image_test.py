"""Detect objects in random test images (boxes + class labels + confidence).

    python image_test.py                         # 5 random images
    python image_test.py --n 10 --conf 0.3
    python image_test.py --weights runs/exp_ep50/weights/best.pt --show

Annotated images are saved to runs/image_test/ (--show also opens a window).
"""
import argparse
import random

import cv2
from ultralytics import YOLO

from common import IMG_EXT, RUNS, get_device, pick_image_dir, resolve_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--show", action="store_true", help="display windows (needs WSLg / GUI)")
    args = ap.parse_args()

    weights = resolve_weights(args.weights)
    image_dir = pick_image_dir()
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    selected = random.Random(args.seed).sample(images, min(args.n, len(images)))
    print(f"Model: {weights}\nImages from: {image_dir}")

    model = YOLO(str(weights))
    results = model.predict(
        source=[str(p) for p in selected], conf=args.conf, imgsz=args.imgsz,
        device=get_device(), save=True,
        project=str(RUNS), name="image_test", exist_ok=True, verbose=False,
    )

    for path, r in zip(selected, results):
        print(f"\nImage: {path.name}  ({len(r.boxes)} objects)")
        for box in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            print(f"  {model.names[int(box.cls[0])]:14s} conf={float(box.conf[0]):.2f} "
                  f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        if args.show:
            cv2.imshow("YOLO Image Detection (any key = next, q = quit)", r.plot())
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"\nImage testing completed! Saved to {RUNS / 'image_test'}")


if __name__ == "__main__":
    main()
