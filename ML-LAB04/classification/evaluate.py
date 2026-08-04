import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ตั้งค่า font ให้รองรับภาษาไทย (Windows มี Tahoma ติดเครื่องอยู่แล้วโดย default)
plt.rcParams["font.family"] = "Tahoma"
plt.rcParams["axes.unicode_minus"] = False


def compute_accuracy(y_true, y_pred):
    """คำนวณ accuracy = จำนวนที่ทายถูก / จำนวนทั้งหมด"""
    return np.mean(y_true == y_pred)


def build_confusion_matrix(y_true, y_pred, class_labels):
    """
    สร้าง confusion matrix ด้วยมือ (ไม่พึ่ง sklearn.metrics)

    confusion_matrix[i][j] = จำนวนตัวอย่างที่ label จริงเป็น class_labels[i]
                              แต่โมเดลทายว่าเป็น class_labels[j]
    """
    n_classes = len(class_labels)
    label_to_idx = {label: i for i, label in enumerate(class_labels)}

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        i = label_to_idx[true_label]
        j = label_to_idx[pred_label]
        matrix[i, j] += 1

    return matrix


def plot_confusion_matrix(matrix, class_names, save_path):
    """Plot confusion matrix เป็น heatmap แล้วเซฟเป็นไฟล์รูป"""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - KNN Classification")

    # เขียนตัวเลขในแต่ละช่อง
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > matrix.max() / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"บันทึก confusion matrix ไว้ที่: {save_path}")


def plot_k_curve(k_values, accuracies, save_path):
    """Plot กราฟเปรียบเทียบ accuracy ของแต่ละค่า k แล้วเซฟเป็นไฟล์รูป"""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, accuracies, marker="o", linewidth=2, markersize=8)

    for k, acc in zip(k_values, accuracies):
        ax.annotate(f"{acc:.4f}", (k, acc),
                    textcoords="offset points", xytext=(0, 10), ha="center")

    ax.set_xlabel("k (จำนวนเพื่อนบ้าน)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("เปรียบเทียบ Accuracy ตามค่า k")
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"บันทึก k-curve ไว้ที่: {save_path}")


def save_predictions_csv(y_true, y_pred, class_names_map, save_path):
    """เซฟผลการทำนายเทียบกับ label จริงเป็นไฟล์ CSV"""
    df = pd.DataFrame({
        "true_label": y_true,
        "true_class_name": [class_names_map[label] for label in y_true],
        "predicted_label": y_pred,
        "predicted_class_name": [class_names_map[label] for label in y_pred],
        "correct": y_true == y_pred,
    })
    df.to_csv(save_path, index=False)
    print(f"บันทึกผลการทำนายไว้ที่: {save_path}")