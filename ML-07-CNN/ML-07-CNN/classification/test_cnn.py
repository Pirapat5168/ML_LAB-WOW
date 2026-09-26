"""
test_cnn.py
ทดสอบโมเดลที่เทรนเสร็จแล้ว ด้วยการสุ่มภาพจาก test set 4 รูป มาทำนายและแสดงผล
"""
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from cnn_model import load_model, predict


def test_random_samples(model_path="outputs/cnn_model.keras",
                         x_test_path="outputs/X_test.npy",
                         y_test_path="outputs/y_test.npy",
                         classes_path="outputs/classes.json",
                         output_dir="outputs",
                         n_samples=4):
    model = load_model(model_path)
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    with open(classes_path) as f:
        classes = json.load(f)

    idx = random.sample(range(len(X_test)), n_samples)
    X_sample = X_test[idx]
    y_true = y_test[idx]

    preds, probs = predict(model, X_sample)

    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
    for i, ax in enumerate(axes):
        ax.imshow(X_sample[i])
        true_label = classes[y_true[i]]
        pred_label = classes[preds[i]]
        conf = float(probs[i]) if probs.ndim == 1 else float(probs[i].max())
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.2f})", color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_sample.png", dpi=150)
    plt.close()
    print(f"[test_cnn] บันทึกผลทำนายตัวอย่างที่ {output_dir}/prediction_sample.png")


if __name__ == "__main__":
    test_random_samples()
