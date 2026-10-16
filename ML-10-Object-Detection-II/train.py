"""Train YOLOv8 with different epochs / image sizes / batch sizes.

    python train.py                       # all 5 experiments (very long!)
    python train.py --exp ep10 ep30       # only some of them
    python train.py --fraction 0.2        # use 20 % of train images (faster)

Finished experiments are skipped; interrupted ones are resumed automatically.
Outputs: runs/exp_<name>/ (weights, results.csv, plots) and
         runs/summary/training_log.csv (time per experiment).
"""
import argparse
import csv
import time

from ultralytics import YOLO

from common import (DATA_YAML, EXPERIMENTS, RUNS, SUMMARY, epochs_done,
                    exp_dir, get_device)


def train_one(name, cfg, args, device):
    out = exp_dir(name)
    last = out / "weights" / "last.pt"
    done = epochs_done(name)

    if done >= cfg["epochs"] and not args.force:
        print(f"[SKIP] {name}: already finished ({done} epochs). Use --force to retrain.")
        return None

    print(f"\n===== Training {name}: {cfg} fraction={args.fraction} =====")
    t0 = time.time()

    if last.exists() and 0 < done < cfg["epochs"] and not args.force:
        print(f"[RESUME] {name} from epoch {done}")
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(args.model)                      # pretrained weights
        model.train(
            data=str(DATA_YAML),
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=device,
            workers=args.workers,
            fraction=args.fraction,
            amp=True,
            cache=False,
            seed=0,
            project=str(RUNS),                        # absolute -> no runs/detect/runs
            name=f"exp_{name}",
            exist_ok=True,
            save=True,
            plots=True,
        )

    minutes = (time.time() - t0) / 60
    print(f"[DONE] {name}: {minutes:.1f} min | best: {out / 'weights' / 'best.pt'}")
    return minutes


def log_time(name, cfg, fraction, minutes):
    SUMMARY.mkdir(parents=True, exist_ok=True)
    f = SUMMARY / "training_log.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["experiment", "epochs", "imgsz", "batch", "fraction", "minutes"])
        w.writerow([name, cfg["epochs"], cfg["imgsz"], cfg["batch"], fraction, f"{minutes:.1f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["all"], choices=["all", *EXPERIMENTS])
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers (WSL: 4-8)")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use")
    ap.add_argument("--force", action="store_true", help="retrain even if finished")
    args = ap.parse_args()

    names = list(EXPERIMENTS) if "all" in args.exp else args.exp
    device = get_device()

    for name in names:
        minutes = train_one(name, EXPERIMENTS[name], args, device)
        if minutes is not None:
            log_time(name, EXPERIMENTS[name], args.fraction, minutes)

    print("\nTraining completed! Next: python evaluate.py && python plot_results.py")


if __name__ == "__main__":
    main()
