
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, classes, cm_path, report_path=None):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print(report)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Test accuracy: {acc:.4f}\n\n{report}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), labels=classes)
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    return acc


def plot_training_history(histories, path):
    n = len(histories)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n), squeeze=False)
    for row, (name, h) in zip(axes, histories.items()):
        epochs = range(1, len(h["accuracy"]) + 1)
        row[0].plot(epochs, h["accuracy"], label="train")
        row[0].plot(epochs, h["val_accuracy"], label="validation")
        row[0].set_title(f"{name} - Accuracy")
        row[1].plot(epochs, h["loss"], label="train")
        row[1].plot(epochs, h["val_loss"], label="validation")
        row[1].set_title(f"{name} - Loss")
        for ax in row:
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_epoch_comparison(results, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in results.items():
        eps = sorted(int(e) for e in r["by_epoch"])
        ax.plot(eps, [r["by_epoch"][str(e)]["test_acc"] for e in eps], marker="o", label=name)
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. epochs for each DCNN configuration")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
def plot_overlay_comparison(histories, path):
    plt.style.use("seaborn-v0_8-darkgrid")
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Training Performance Comparison", fontsize=16, fontweight="bold")

    for i, (name, h) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(h["accuracy"]) + 1)
        final_val_acc = h["val_accuracy"][-1] * 100

        line_style = dict(marker="o", markersize=5, markeredgecolor="white",
                          markeredgewidth=0.6, linewidth=2.2, color=color)
        label = f"{name} ({final_val_acc:.2f}%)"

        axes[0, 0].plot(epochs, [a * 100 for a in h["accuracy"]], label=label, **line_style)
        axes[0, 1].plot(epochs, [a * 100 for a in h["val_accuracy"]], label=label, **line_style)
        axes[1, 0].plot(epochs, h["loss"], label=label, **line_style)
        axes[1, 1].plot(epochs, h["val_loss"], label=label, **line_style)

    titles = [
        ("(a) Training Accuracy of Models", "Epochs", "Accuracy (%)"),
        ("(b) Validation Accuracy of Models", "Epochs", "Accuracy (%)"),
        ("(c) Training Loss of Models", "Epochs", "Loss"),
        ("(d) Validation Loss of Models", "Epochs", "Loss"),
    ]
    for ax, (title, xlabel, ylabel) in zip(axes.flat, titles):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 1].set_ylim(0, 100)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)