import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = {
    1: "Mammal",
    2: "Bird",
    3: "Reptile",
    4: "Fish",
    5: "Amphibian",
    6: "Insect",
    7: "Invertebrate",
}


def load_raw_data(csv_path="../data-animal/animal_dataset.csv"):
    """โหลดไฟล์ CSV ดิบเข้ามาเป็น pandas DataFrame"""
    df = pd.read_csv(csv_path)
    return df


def load_and_prepare_data(csv_path="../data-animal/animal_dataset.csv"):
    """
    โหลดข้อมูลและเตรียมสำหรับ clustering

    Returns
    -------
    X_scaled       : numpy array ของ feature ที่ standardize แล้ว (ใช้เทรน K-Means)
    animal_names   : ชื่อสัตว์แต่ละแถว (เรียงตรงกับ X_scaled)
    true_labels    : class_type จริง (เก็บไว้ใช้ "เทียบผลทีหลัง" เท่านั้น
                      ห้ามส่งเข้าไปในขั้นตอนเทรน K-Means)
    feature_names  : รายชื่อคอลัมน์ feature ทั้งหมด
    scaler         : StandardScaler ที่ fit แล้ว
    """
    df = load_raw_data(csv_path)

    feature_names = [c for c in df.columns if c not in ("animal_name", "class_type")]

    X = df[feature_names].values.astype(np.float32)
    animal_names = df["animal_name"].values
    true_labels = df["class_type"].values.astype(np.int32)

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, animal_names, true_labels, feature_names, scaler


if __name__ == "__main__":
    X_scaled, animal_names, true_labels, feature_names, scaler = load_and_prepare_data()

    print("จำนวนตัวอย่างทั้งหมด:", X_scaled.shape[0])
    print("จำนวน features      :", X_scaled.shape[1])
    print()
    print("ตัวอย่างชื่อสัตว์ 5 ตัวแรก:", animal_names[:5])
    print("ตัวอย่างข้อมูล X แถวแรก (หลัง standardize):")
    print(X_scaled[0])
    print()
    print("class_type จริง (เก็บไว้เทียบทีหลัง) 5 ตัวแรก:", true_labels[:5])