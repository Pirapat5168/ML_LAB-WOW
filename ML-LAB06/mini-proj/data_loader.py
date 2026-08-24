"""
data_loader.py
---------------
โหลด dataset สำหรับ LAB 6 (Neural Network)

ใช้ dataset เดิมจาก LAB 3 (age_gender / UTKFace ที่ทำ PCA ไว้แล้ว
เหลือ 50 principal components: pc_1 ... pc_50) เพื่อทำนาย "gender"
(0 = Male, 1 = Female) ด้วย Neural Network แทนที่ Logistic Regression

ไฟล์ที่ใช้: data/features_pca.csv
คอลัมน์: filename, age, gender, pc_1 ... pc_50
"""

import pandas as pd
import numpy as np

DATA_PATH = "data/features_pca.csv"


def load_data(path: str = DATA_PATH):
    """
    โหลดข้อมูลจากไฟล์ csv แล้วแยกเป็น X (features) และ y (label)

    Returns
    -------
    X : np.ndarray  shape (n_samples, 50)   -> pc_1 ... pc_50
    y : np.ndarray  shape (n_samples,)      -> gender (0=Male, 1=Female)
    df : pd.DataFrame  ข้อมูลดิบทั้งหมด (เผื่อใช้ดู filename/age เพิ่มเติม)
    """
    df = pd.read_csv(path)

    pc_cols = [c for c in df.columns if c.startswith("pc_")]
    if len(pc_cols) == 0:
        raise ValueError("ไม่พบคอลัมน์ pc_1..pc_50 ในไฟล์ข้อมูล กรุณาตรวจสอบ path ของ dataset")

    X = df[pc_cols].values.astype(np.float32)
    y = df["gender"].values.astype(np.int32)

    return X, y, df


if __name__ == "__main__":
    X, y, df = load_data()
    print(f"จำนวนข้อมูลทั้งหมด : {len(df)}")
    print(f"จำนวน feature (PCA): {X.shape[1]}")
    print(f"Gender distribution : {pd.Series(y).value_counts().to_dict()}  (0=Male, 1=Female)")
