
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cnn_model import predict


def evaluate_model(model, X_test, y_test, classes, output_dir="outputs"):
    """คำนวณ accuracy + classification report + confusion matrix, save รูปและผล"""
    preds, probs = predict(model, X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=classes, digits=4)

    print(f"\n[evaluate] Test Accuracy: {acc:.4f}")
    print(report)

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    return {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist()}


def plot_training_history(history, output_dir="outputs", filename="training_history.png"):
    """plot กราฟ accuracy/loss ของ train vs validation ต่อ epoch แล้ว save history.json"""
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(hist["accuracy"], label="train")
    axes[0].plot(hist["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist["loss"], label="train")
    axes[1].plot(hist["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=150)
    plt.close()

    with open(f"{output_dir}/history.json", "w") as f:
        json.dump(hist, f, indent=2)
def plot_comparison_curves(all_histories, output_dir="outputs",
                            filename="comparison_training_performance.png"):
    """
    plot กราฟเปรียบเทียบทุก config บนกราฟเดียวกัน (เส้นสีต่างกัน = config ต่างกัน)
    แบบเดียวกับกราฟ Training Performance ในงานวิจัย (4 กราฟ: train acc, val acc,
    train loss, val loss)

    all_histories: dict ของ {config_name: history.history (dict)}
                    เช่น {"cfg1_conv2_dense64_ep10": history.history, ...}
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    colors = plt.cm.tab10.colors  # ชุดสีสดใส แยกจากกันชัดเจน สูงสุด 10 เส้น

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Performance Comparison", fontsize=16, fontweight="bold")

    for i, (name, hist) in enumerate(all_histories.items()):
        color = colors[i % len(colors)]
        best_val_acc = max(hist["val_accuracy"]) * 100
        label = f"{name} (best: {best_val_acc:.2f}%)"

        axes[0, 0].plot(hist["accuracy"], label=label, color=color, linewidth=2.2)
        axes[0, 1].plot(hist["val_accuracy"], label=label, color=color, linewidth=2.2)
        axes[1, 0].plot(hist["loss"], label=label, color=color, linewidth=2.2)
        axes[1, 1].plot(hist["val_loss"], label=label, color=color, linewidth=2.2)

    titles = [
        ("(a) Training accuracy of models", "Epoch", "Accuracy"),
        ("(b) Validation accuracy of models", "Epoch", "Accuracy"),
        ("(c) Training loss of models", "Epoch", "Loss"),
        ("(d) Validation loss of models", "Epoch", "Loss"),
    ]
    for ax, (title, xlabel, ylabel) in zip(axes.flat, titles):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{output_dir}/{filename}", dpi=200)
    plt.close()
    print(f"[evaluate] บันทึกกราฟเปรียบเทียบทุก config ที่ {output_dir}/{filename}")