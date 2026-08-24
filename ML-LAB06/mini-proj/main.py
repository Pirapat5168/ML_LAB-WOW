"""
main.py
--------
LAB 6: Neural Network และการประยุกต์ใช้งาน
รันทุกขั้นตอนตามโจทย์:
  1. เลือกและโหลด dataset (age_gender / UTKFace ผ่าน PCA จาก LAB 3)
  2. แบ่ง train/test
  3. Standardize input features
  4. สร้าง Neural Network model
  5. เทรนด้วยจำนวน epoch ต่างกัน (เปรียบเทียบ)
  6. เปรียบเทียบ NN config ต่างกัน (จำนวน hidden layer / neuron)
  7. ประเมินผลด้วย accuracy
  8. บันทึกผลลัพธ์ + กราฟ + คำทำนาย
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from data_loader import load_data
from split_data import split
from preprocessing import standardize
from nn_model import build_model, NN_CONFIGS, EPOCH_OPTIONS
from evaluate import (
    evaluate_model, plot_training_history, plot_confusion_matrix, plot_comparison_bar
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # ---------- 1) โหลดข้อมูล ----------
    X, y, df = load_data()
    print(f"จำนวนข้อมูลทั้งหมด: {len(df)}  | จำนวน feature: {X.shape[1]}")
    print(f"Gender distribution: {pd.Series(y).value_counts().to_dict()} (0=Male, 1=Female)")

    # ---------- 2) แบ่ง train/test ----------
    X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=SEED)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ---------- 3) Standardize ----------
    X_train_s, X_test_s, scaler = standardize(X_train, X_test)

    all_results = []

    # ==================================================================
    # EXPERIMENT 1: เปรียบเทียบจำนวน epoch (ใช้ architecture เดียวกัน)
    # ==================================================================
    base_arch = NN_CONFIGS["Medium_2Layer"]  # [64, 32]
    epoch_results = []

    print("\n" + "#" * 70)
    print("# EXPERIMENT 1: เปรียบเทียบจำนวน Epoch (architecture = 64-32)")
    print("#" * 70)

    for n_epochs in EPOCH_OPTIONS:
        name = f"epoch{n_epochs}"
        model = build_model(input_dim=X_train_s.shape[1], hidden_layers=base_arch)
        history = model.fit(
            X_train_s, y_train,
            validation_split=0.2,
            epochs=n_epochs,
            batch_size=32,
            verbose=0,
        )
        acc, y_pred, y_proba = evaluate_model(model, X_test_s, y_test, name=name)
        plot_training_history(history, name=name)

        epoch_results.append({"epochs": n_epochs, "architecture": "64-32", "accuracy": acc})
        all_results.append({"experiment": "epoch_comparison", "config": name,
                             "epochs": n_epochs, "architecture": "64-32", "accuracy": acc})

    epoch_df = pd.DataFrame(epoch_results)
    plot_comparison_bar(epoch_df, x_col="epochs",
                         title="Accuracy vs Number of Epochs (architecture 64-32)",
                         filename="compare_epochs.png")

    # ==================================================================
    # EXPERIMENT 2: เปรียบเทียบ NN configuration (hidden layers / neurons)
    # ==================================================================
    fixed_epochs = 30
    arch_results = []
    best_acc, best_model, best_name, best_pred = -1, None, None, None
    best_history = None

    print("\n" + "#" * 70)
    print(f"# EXPERIMENT 2: เปรียบเทียบ NN configuration (epoch คงที่ = {fixed_epochs})")
    print("#" * 70)

    for cfg_name, hidden_layers in NN_CONFIGS.items():
        model = build_model(input_dim=X_train_s.shape[1], hidden_layers=hidden_layers)
        history = model.fit(
            X_train_s, y_train,
            validation_split=0.2,
            epochs=fixed_epochs,
            batch_size=32,
            verbose=0,
        )
        acc, y_pred, y_proba = evaluate_model(model, X_test_s, y_test, name=cfg_name)
        plot_training_history(history, name=cfg_name)
        plot_confusion_matrix(y_test, y_pred, name=cfg_name)

        arch_results.append({
            "config": cfg_name,
            "hidden_layers": str(hidden_layers),
            "n_layers": len(hidden_layers),
            "accuracy": acc,
        })
        all_results.append({"experiment": "architecture_comparison", "config": cfg_name,
                             "epochs": fixed_epochs, "architecture": str(hidden_layers),
                             "accuracy": acc})

        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, cfg_name
            best_pred, best_history = y_pred, history

    arch_df = pd.DataFrame(arch_results)
    plot_comparison_bar(arch_df, x_col="config",
                         title=f"Accuracy vs NN Configuration (epoch={fixed_epochs})",
                         filename="compare_architectures.png")

    # ---------- บันทึกโมเดลที่ดีที่สุด ----------
    best_model.save(os.path.join(OUTPUT_DIR, "best_model.keras"))
    print(f"\nโมเดลที่ดีที่สุด: {best_name}  (Accuracy = {best_acc:.4f}) -> บันทึกที่ outputs/best_model.keras")

    # ---------- บันทึก scaler config (mean/scale) ไว้ใช้กับ test_nn.py ----------
    np.save(os.path.join(OUTPUT_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, "scaler_scale.npy"), scaler.scale_)

    # ---------- สรุปผลรวมทุก config ลง CSV ----------
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)
    print(f"\nบันทึกสรุปผลลัพธ์ทั้งหมด: outputs/results_summary.csv")
    print(results_df.to_string(index=False))

    # ---------- Prediction ตัวอย่าง 10 รายการจาก test set (ใช้โมเดลที่ดีที่สุด) ----------
    sample_idx = np.random.RandomState(SEED).choice(len(X_test_s), size=10, replace=False)
    sample_proba = best_model.predict(X_test_s[sample_idx], verbose=0).ravel()
    sample_pred = (sample_proba >= 0.5).astype(int)
    label_map = {0: "Male", 1: "Female"}

    pred_df = pd.DataFrame({
        "actual": [label_map[v] for v in y_test[sample_idx]],
        "predicted": [label_map[v] for v in sample_pred],
        "probability_female": np.round(sample_proba, 3),
        "correct": y_test[sample_idx] == sample_pred,
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "sample_predictions.csv"), index=False)
    print(f"\nตัวอย่างการทำนาย 10 รายการ (โมเดล {best_name}):")
    print(pred_df.to_string(index=False))
    print(f"\nบันทึกตัวอย่างการทำนาย: outputs/sample_predictions.csv")


if __name__ == "__main__":
    main()
