"""
test_nn.py
-----------
โหลดโมเดลที่เทรนไว้แล้ว (outputs/best_model.keras) มาทดสอบทำนาย
กับข้อมูลจริงจาก dataset (จำลองการนำโมเดลไปใช้งานจริง)

รันหลังจาก main.py เสร็จแล้วเท่านั้น (ต้องมี outputs/best_model.keras)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from data_loader import load_data

OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.keras")


def load_scaler():
    mean = np.load(os.path.join(OUTPUT_DIR, "scaler_mean.npy"))
    scale = np.load(os.path.join(OUTPUT_DIR, "scaler_scale.npy"))
    return mean, scale


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "ไม่พบ outputs/best_model.keras กรุณารัน main.py ก่อน เพื่อเทรนและบันทึกโมเดล"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    mean, scale = load_scaler()

    X, y, df = load_data()

    # สุ่มตัวอย่าง 5 รายการมาทดสอบ (จำลองข้อมูลใหม่ที่เพิ่งเข้ามา)
    rng = np.random.RandomState(7)
    idx = rng.choice(len(X), size=5, replace=False)

    X_new = X[idx]
    X_new_scaled = (X_new - mean) / scale  # standardize ด้วย scaler ที่บันทึกไว้จากตอนเทรน

    proba = model.predict(X_new_scaled, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    label_map = {0: "Male", 1: "Female"}

    result = pd.DataFrame({
        "filename": df.iloc[idx]["filename"].values,
        "actual": [label_map[v] for v in y[idx]],
        "predicted": [label_map[v] for v in pred],
        "probability_female": np.round(proba, 3),
    })

    print("ผลการทดสอบโมเดลกับข้อมูลตัวอย่างใหม่:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
