
import os
import json
import numpy as np

from data_loader import load_image_paths
from preprocessing import load_and_preprocess
from split_data import split_dataset
from cnn_model import build_cnn, train_cnn, save_model
from evaluate import evaluate_model, plot_training_history, plot_comparison_curves

DATASET_DIR = "../DATASET"
OUTPUT_DIR = "outputs"

# Config ที่จะเปรียบเทียบกัน ตามโจทย์: จำนวน conv layer, จำนวน neuron, จำนวน epoch
CONFIGS = [
    {"name": "cfg1_conv2_dense64_ep10", "conv_layers": 2, "dense_neurons": 64, "epochs": 10},
    {"name": "cfg2_conv3_dense128_ep10", "conv_layers": 3, "dense_neurons": 128, "epochs": 10},
    {"name": "cfg3_conv3_dense128_ep20", "conv_layers": 3, "dense_neurons": 128, "epochs": 20},
    {"name": "cfg4_conv4_dense256_ep20", "conv_layers": 4, "dense_neurons": 256, "epochs": 20},
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) โหลด path รูปภาพ + label
    image_paths, labels, classes = load_image_paths(DATASET_DIR)

    # 2) preprocess: resize + BGR->RGB + normalize
    X, valid_idx = load_and_preprocess(image_paths)
    y = np.array([labels[i] for i in valid_idx])

    np.save(f"{OUTPUT_DIR}/features.npy", X)
    np.save(f"{OUTPUT_DIR}/labels.npy", y)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    # 3) split train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
                       ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        np.save(f"{OUTPUT_DIR}/{name}.npy", arr)

    input_shape = X_train.shape[1:]
    num_classes = len(classes)

    # 4) เทรนแต่ละ config เปรียบเทียบกัน
    results = []
    all_histories = {}
    best_acc = -1
    best_model = None
    best_history = None
    best_name = None

    for cfg in CONFIGS:
        print(f"\n{'=' * 60}\n[main] กำลังเทรน config: {cfg['name']}\n{'=' * 60}")
        model = build_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            conv_layers=cfg["conv_layers"],
            dense_neurons=cfg["dense_neurons"],
        )
        history = train_cnn(model, X_train, y_train, X_val, y_val, epochs=cfg["epochs"])
        all_histories[cfg["name"]] = history.history

        val_acc = max(history.history["val_accuracy"])

        results.append({
            "name": cfg["name"],
            "conv_layers": cfg["conv_layers"],
            "dense_neurons": cfg["dense_neurons"],
            "epochs": cfg["epochs"],
            "best_val_accuracy": val_acc,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_history = history
            best_name = cfg["name"]

        # เซฟโมเดล + history ของ config นี้ทันที กันข้อมูลหายถ้า error ระหว่างทาง
        save_model(model, f"{OUTPUT_DIR}/model_{cfg['name']}.keras")
        with open(f"{OUTPUT_DIR}/history_{cfg['name']}.json", "w") as f:
            json.dump(history.history, f, indent=2)
        print(f"[main] เซฟ checkpoint ของ {cfg['name']} แล้ว (กันเหตุฉุกเฉิน)")

    # 5) บันทึกผลเปรียบเทียบทุก config
    # 5) บันทึกผลเปรียบเทียบทุก config
    with open(f"{OUTPUT_DIR}/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[main] สรุปผลเปรียบเทียบ config:")
    for r in results:
        print(f"  {r['name']}: val_accuracy={r['best_val_accuracy']:.4f}")
    print(f"\n[main] Config ที่ดีที่สุด: {best_name} (val_accuracy={best_acc:.4f})")

    # 6) evaluate โมเดลที่ดีที่สุดบน test set + save กราฟ/confusion matrix
    evaluate_model(best_model, X_test, y_test, classes, output_dir=OUTPUT_DIR)
    plot_training_history(best_history, output_dir=OUTPUT_DIR)

    # plot กราฟเปรียบเทียบทุก config บนกราฟเดียวกัน (แบบที่อาจารย์ต้องการ)
    plot_comparison_curves(all_histories, output_dir=OUTPUT_DIR)
    # 7) save โมเดลที่ดีที่สุด
    save_model(best_model, f"{OUTPUT_DIR}/cnn_model.keras")


if __name__ == "__main__":
    main()
