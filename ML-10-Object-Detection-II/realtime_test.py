"""Real-time detection from a camera (or any stream). Press Q to exit.

    python realtime_test.py                      # camera index 0
    python realtime_test.py --source 1           # other camera
    python realtime_test.py --source http://PHONE_IP:8080/video   # IP-camera app
    python realtime_test.py --save               # also record to runs/realtime_test/

NOTE (WSL2): WSL does not see USB webcams by default. Either attach the camera
with usbipd-win, use an IP-camera/stream URL, or run this script with the
Windows Python (copy best.pt, same code). See README.md.
"""
import argparse
import time

import cv2
from ultralytics import YOLO

from common import RUNS, get_device, resolve_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or stream URL")
    ap.add_argument("--weights")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", action="store_true", help="record annotated output")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    args = ap.parse_args()

    model = YOLO(str(resolve_weights(args.weights)))
    device = get_device()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open source '{args.source}'. On WSL2 the webcam is not "
            "available by default - see README (usbipd / IP camera / Windows Python).")

    writer = None
    if args.save:
        out_dir = RUNS / "realtime_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = out_dir / f"realtime_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        print(f"Recording to {out}")

    prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                              device=device, verbose=False)[0]
            annotated = r.plot()

            now = time.time()
            fps, prev = 1.0 / max(now - prev, 1e-6), now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if writer:
                writer.write(annotated)
            if not args.no_show:
                cv2.imshow("YOLO Real-time Detection (q = exit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
