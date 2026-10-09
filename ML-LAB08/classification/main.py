
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from data_loader import load_dataset
from evaluate import evaluate_model, plot_epoch_comparison, plot_training_history, plot_overlay_comparison
from preprocessing import compute_channel_stats
from split_data import load_splits, save_splits, split_data
from vgg_model import CONFIGS, build_vgg, save_model, train_vgg

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT.parent / "PetImages"))
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--max-per-class", type=int, default=5000, help="0 = use all images")
    p.add_argument("--epochs", type=int, nargs="+", default=[5, 10, 15],
                   help="epoch counts to compare; training runs up to the largest")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--reuse-data", action="store_true",
                   help="reuse saved .npy files in outputs/ instead of reloading images")
    return p.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(exist_ok=True)
    keras.utils.set_random_seed(42)
    final_epoch = max(args.epochs)

    # 1-2) Load + split ---------------------------------------------------
    if args.reuse_data and (OUT / "X_train.npy").exists():
        print("Reusing saved data from outputs/")
        splits = load_splits(OUT)
        classes = json.loads((OUT / "classes.json").read_text(encoding="utf-8"))
    else:
        X, y, classes = load_dataset(args.data_dir, args.img_size, args.max_per_class or None)
        np.save(OUT / "labels.npy", y)
        (OUT / "classes.json").write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
        splits = split_data(X, y)
        save_splits(OUT, *splits)
        del X
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    print(f"Classes: {classes}")
    print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

    # 3) Standardization stats (from the training set only) ---------------
    mean, std = compute_channel_stats(X_train)
    print(f"Channel mean {mean}, std {std}")

    # 4) Train every configuration ---------------------------------------
    results, histories = {}, {}
    best_name, best_val = None, -1.0
    for name in args.configs:
        print(f"\n{'=' * 60}\nConfiguration: {name}  {CONFIGS[name]}\n{'=' * 60}")
        model = build_vgg(CONFIGS[name], X_train.shape[1:], len(classes), mean, std)
        history, by_epoch, secs = train_vgg(model, splits, args.epochs, args.batch_size)
        histories[name] = history
        results[name] = {
            "config": CONFIGS[name],
            "params": int(model.count_params()),
            "train_time_s": round(secs, 1),
            "by_epoch": {str(k): v for k, v in sorted(by_epoch.items())},
        }
        val_final = history["val_accuracy"][-1]
        if val_final > best_val:                      # best config chosen by VALIDATION accuracy
            best_val, best_name = val_final, name
            save_model(model, OUT / "vgg_model.keras")
        keras.backend.clear_session()

    # 5) Save results ------------------------------------------------------
    (OUT / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    (OUT / "comparison_results.json").write_text(
        json.dumps({"best_config": best_name, "results": results}, indent=2), encoding="utf-8")
    with open(OUT / "comparison_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "epochs", "train_acc", "val_acc", "test_acc", "params", "train_time_s"])
        for name, r in results.items():
            for ep, v in r["by_epoch"].items():
                w.writerow([name, ep, f"{v['train_acc']:.4f}", f"{v['val_acc']:.4f}",
                            f"{v['test_acc']:.4f}", r["params"], r["train_time_s"]])

    print(f"\n{'=' * 60}\nACCURACY COMPARISON (test accuracy)\n{'=' * 60}")
    header = f"{'Config':<18}" + "".join(f"{'ep ' + str(e):>9}" for e in sorted(args.epochs))
    print(header)
    for name, r in results.items():
        print(f"{name:<18}" + "".join(
            f"{r['by_epoch'][str(e)]['test_acc']:>9.4f}" for e in sorted(args.epochs)))
    print(f"\nBest configuration (by validation accuracy @ {final_epoch} epochs): {best_name}")

    plot_training_history(histories, OUT / "training_history.png")
    plot_epoch_comparison(results, OUT / "comparison_epochs.png")
    plot_overlay_comparison(histories, OUT / "comparison_overlay.png")

    # 6) Final evaluation of the best model on the test set ---------------
    best_model = keras.models.load_model(OUT / "vgg_model.keras")
    evaluate_model(best_model, X_test, y_test, classes,
                   cm_path=OUT / "confusion_matrix.png",
                   report_path=OUT / "classification_report.txt")
    print("\nDone. Next: python test_vgg.py   (predictions on 4 random test images)")


if __name__ == "__main__":
    main()
