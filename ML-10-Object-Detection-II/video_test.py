"""Detect objects in a video file and save the annotated video.

    python video_test.py --source my_video.mp4
    python video_test.py --source my_video.mp4 --weights runs/exp_ep50/weights/best.pt --show

Output: runs/video_test/<video name>_detected.mp4
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import RUNS, get_device, resolve_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path to a video file")
    ap.add_argument("--weights")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--show", action="store_true", help="display while processing (q = stop)")
    args = ap.parse_args()

    src = Path(args.source).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")

    model = YOLO(str(resolve_weights(args.weights)))
    device = get_device()

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = RUNS / "video_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    n, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                              device=device, verbose=False)[0]
            annotated = r.plot()
            writer.write(annotated)
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{total} frames | {n / (time.time() - t0):.1f} FPS")
            if args.show:
                cv2.imshow("YOLO Video Detection (q = stop)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Processed {n} frames at {n / max(time.time() - t0, 1e-6):.1f} FPS")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
