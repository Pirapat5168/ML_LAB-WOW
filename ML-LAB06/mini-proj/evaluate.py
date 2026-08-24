"""
evaluate.py
------------
ประเมินผล Neural Network: accuracy, confusion matrix,
กราฟ training/validation accuracy & loss, และกราฟเปรียบเทียบ
ระหว่าง config / epoch ต่าง ๆ
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

OUTPUT_DIR = "outputs"


def evaluate_model(model, X_test, y_test, name: str):
    """คืนค่า accuracy พร้อม print classification report"""
    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Male", "Female"]))

    return acc, y_pred, y_proba


def plot_training_history(history, name: str, out_dir: str = OUTPUT_DIR):
    """plot กราฟ accuracy และ loss ของ training/validation ต่อ epoch"""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title(f"Accuracy - {name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title(f"Loss - {name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"history_{name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"บันทึกกราฟ: {path}")


def plot_confusion_matrix(y_test, y_pred, name: str, out_dir: str = OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"confusion_matrix_{name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"บันทึกกราฟ: {path}")


def plot_comparison_bar(results_df: pd.DataFrame, x_col: str, title: str,
                         filename: str, out_dir: str = OUTPUT_DIR):
    """สร้างกราฟแท่งเปรียบเทียบ accuracy ระหว่าง config หรือ epoch ต่าง ๆ"""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    bars = plt.bar(results_df[x_col].astype(str), results_df["accuracy"], color="#4C72B0")
    for b, acc in zip(bars, results_df["accuracy"]):
        plt.text(b.get_x() + b.get_width()/2, acc + 0.005, f"{acc:.3f}",
                  ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"บันทึกกราฟ: {path}")
