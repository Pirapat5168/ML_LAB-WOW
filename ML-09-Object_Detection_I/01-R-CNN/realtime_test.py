"""Webcam demo of the R-CNN detector. Press Q to quit.

    python realtime_test.py --camera 0 --threshold 0.85 --regions 25
(CPU-friendly: fewer regions = faster.)
"""
from pathlib import Path
import sys
import time
import argparse
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import draw   # noqa: E402
from train import load_model, detect   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--regions", type=int, default=25)
    a = p.parse_args()

    model = load_model()
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {a.camera}")
    print("Press Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            dets = detect(model, frame, a.regions, a.threshold)
            out = draw(frame, dets)
            cv2.putText(out, f"{1 / max(time.time() - t0, 1e-6):.1f} FPS", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("R-CNN - real vs fake face (Q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
