"""Create report figures from the training logs (no GPU needed).

    python plot_results.py

Outputs in runs/summary/:
  loss_curves.png         train / val box, cls, dfl loss per epoch
  metrics_per_epoch.png   P, R, mAP50, mAP50-95 per epoch (validation)
  compare_epochs.png      ep10 vs ep30 vs ep50
  compare_configs.png     ep30 vs img640_b8 vs img480_b16
  results_table.md        final-metrics table (paste into the report)
"""
import matplotlib
matplotlib.use("Agg")                      # headless (WSL)
import matplotlib.pyplot as plt
import pandas as pd

from common import EXPERIMENTS, SUMMARY, exp_dir

LOSSES = [("train/box_loss", "Train box loss"), ("train/cls_loss", "Train cls loss"),
          ("train/dfl_loss", "Train dfl loss"), ("val/box_loss", "Val box loss"),
          ("val/cls_loss", "Val cls loss"), ("val/dfl_loss", "Val dfl loss")]
METRICS = [("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall"),
           ("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]
FINAL = ["precision", "recall", "mAP50", "mAP50-95"]


def load_logs():
    logs = {}
    for name in EXPERIMENTS:
        f = exp_dir(name) / "results.csv"
        if f.exists():
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            logs[name] = df
    return logs


def grid(logs, items, ncols, fname, title):
    nrows = -(-len(items) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    for ax, (col, label) in zip(axes.flat, items):
        for name, df in logs.items():
            ax.plot(df["epoch"], df[col], marker="o", ms=2, label=name)
        ax.set_title(label); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    axes.flat[0].legend()
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(SUMMARY / fname, dpi=150); plt.close(fig)


def final_table(logs):
    """Test-set results if evaluate.py was run, else last-epoch validation results."""
    ev = SUMMARY / "evaluation_summary.csv"
    if ev.exists():
        df = pd.read_csv(ev).set_index("experiment")[FINAL]
        src = "evaluation split (evaluate.py)"
    else:
        rows = {n: [d[c].iloc[-1] for c, _ in METRICS] for n, d in logs.items()}
        df = pd.DataFrame(rows, index=FINAL).T
        src = "validation, last epoch (run evaluate.py for test results)"
    return df, src


def bars(df, names, fname, title):
    names = [n for n in names if n in df.index]
    if len(names) < 2:
        return
    ax = df.loc[names, FINAL].T.plot(kind="bar", figsize=(8, 4.5), rot=0)
    ax.set_title(title); ax.set_ylabel("score"); ax.grid(axis="y", alpha=.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7)
    ax.figure.tight_layout()
    ax.figure.savefig(SUMMARY / fname, dpi=150); plt.close(ax.figure)


# ---------- IEEE-style figures (like paper Fig. "Training Performance") ----------
COLORS = ["#b000b5", "#e41a1c", "#2ca02c", "#1f4fff", "#111111", "#a6d800"]
MARKERS = ["o", "s", "^", "P", "*", "D"]
STYLE = {"font.family": "serif", "font.size": 10, "axes.grid": True,
         "grid.alpha": .35, "axes.facecolor": "#f4f4fb", "figure.dpi": 100}


def add_total_loss(logs):
    for df in logs.values():
        df["train/total_loss"] = df[["train/box_loss", "train/cls_loss", "train/dfl_loss"]].sum(axis=1)
        df["val/total_loss"] = df[["val/box_loss", "val/cls_loss", "val/dfl_loss"]].sum(axis=1)


def panel(ax, logs, col, title, ylabel, pct=False, lower_better=False, legend_loc="lower right"):
    """One curve panel: marker line per experiment, final value in the legend,
    star on the best epoch."""
    for i, (name, df) in enumerate(logs.items()):
        y = df[col] * (100 if pct else 1)
        c, m = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
        fmt = "{:.2f}%" if pct else "{:.4f}"
        ax.plot(df["epoch"], y, marker=m, ms=4, lw=1.4, color=c,
                label=f"{name}: {fmt.format(y.iloc[-1])}")
        b = y.idxmin() if lower_better else y.idxmax()
        ax.plot(df["epoch"][b], y[b], marker="*", ms=13, color=c, mec="black", mew=.8, zorder=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epochs"); ax.set_ylabel(ylabel)
    if pct:
        ax.set_ylim(0, 100)
    ax.legend(loc=legend_loc, fontsize=8, framealpha=.9, title="final (* = best)", title_fontsize=7)


def ieee_figures(logs, tag=""):
    add_total_loss(logs)
    with plt.rc_context(STYLE):
        # Figure A: accuracy-like metrics + loss, 2 x 2 (same layout as the paper)
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        panel(ax[0, 0], logs, "metrics/mAP50(B)", "(a) Validation mAP@0.5", "mAP50 (%)", pct=True)
        panel(ax[0, 1], logs, "metrics/mAP50-95(B)", "(b) Validation mAP@0.5:0.95", "mAP50-95 (%)", pct=True)
        panel(ax[1, 0], logs, "train/total_loss", "(c) Training loss (box+cls+dfl)", "Loss",
              lower_better=True, legend_loc="upper right")
        panel(ax[1, 1], logs, "val/total_loss", "(d) Validation loss (box+cls+dfl)", "Loss",
              lower_better=True, legend_loc="upper right")
        fig.suptitle(f"YOLOv8 Training Performance{tag}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(SUMMARY / f"ieee_performance{('_' + tag.strip(' ()').replace(' ', '_')) if tag else ''}.png", dpi=300)
        plt.close(fig)

        # Figure B: precision / recall + per-term validation loss
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        panel(ax[0, 0], logs, "metrics/precision(B)", "(a) Precision", "Precision (%)", pct=True)
        panel(ax[0, 1], logs, "metrics/recall(B)", "(b) Recall", "Recall (%)", pct=True)
        panel(ax[0, 2], logs, "train/box_loss", "(c) Train box loss", "Loss",
              lower_better=True, legend_loc="upper right")
        panel(ax[1, 0], logs, "val/box_loss", "(d) Val box loss", "Loss",
              lower_better=True, legend_loc="upper right")
        panel(ax[1, 1], logs, "val/cls_loss", "(e) Val cls loss", "Loss",
              lower_better=True, legend_loc="upper right")
        panel(ax[1, 2], logs, "val/dfl_loss", "(f) Val dfl loss", "Loss",
              lower_better=True, legend_loc="upper right")
        fig.suptitle(f"YOLOv8 Detailed Metrics{tag}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(SUMMARY / f"ieee_detail{('_' + tag.strip(' ()').replace(' ', '_')) if tag else ''}.png", dpi=300)
        plt.close(fig)


def per_class_chart():
    """Grouped bars of AP50 per class for every evaluated experiment."""
    files = sorted(SUMMARY.glob("per_class_*.csv"))
    if not files:
        return
    data = {f.stem.replace("per_class_", ""): pd.read_csv(f).set_index("class")["AP50"] for f in files}
    df = pd.DataFrame(data)
    with plt.rc_context(STYLE):
        ax = df.plot(kind="bar", figsize=(10, 5), rot=0, color=COLORS[:len(df.columns)], edgecolor="black", lw=.5)
        ax.set_title("Per-class AP@0.5 (evaluation split)", fontweight="bold")
        ax.set_ylabel("AP50"); ax.set_ylim(0, 1.05)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
        ax.figure.tight_layout()
        ax.figure.savefig(SUMMARY / "per_class_ap50.png", dpi=300)
        plt.close(ax.figure)


def main():
    logs = load_logs()
    if not logs:
        raise SystemExit("No runs/exp_*/results.csv found. Run train.py first.")
    SUMMARY.mkdir(parents=True, exist_ok=True)

    ieee_figures(logs)                                     # all experiments
    groups = {"epochs": ["ep10", "ep30", "ep50"],
              "configs": ["ep30", "img640_b8", "img480_b16"]}
    for tag, names in groups.items():                      # one figure per comparison
        sub = {n: logs[n] for n in names if n in logs}
        if len(sub) >= 2:
            ieee_figures(sub, f" ({tag} comparison)")
    per_class_chart()
    grid(logs, LOSSES, 3, "loss_curves.png", "Training / validation loss")
    grid(logs, METRICS, 2, "metrics_per_epoch.png", "Validation metrics per epoch")

    table, src = final_table(logs)
    bars(table, ["ep10", "ep30", "ep50"], "compare_epochs.png", "Effect of number of epochs")
    bars(table, ["ep30", "img640_b8", "img480_b16"], "compare_configs.png",
         "Effect of image size / batch size (30 epochs)")

    md = f"Source: {src}\n\n" + table.round(4).to_markdown()
    (SUMMARY / "results_table.md").write_text(md + "\n")
    print(md)
    print(f"\nFigures saved in {SUMMARY}")


if __name__ == "__main__":
    main()