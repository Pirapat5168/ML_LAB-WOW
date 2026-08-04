import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ชื่อ class_type -> ชื่อสัตว์ (สำหรับแสดงผลให้อ่านง่าย)
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


def load_and_prepare_data(csv_path="../data-animal/animal_dataset.csv",
                           test_size=0.2,
                           random_state=42):
    """
    โหลดข้อมูล, แยก feature กับ label, แบ่ง train/test, และ standardize

    Returns
    -------
    X_train, X_test : numpy array ของ feature ที่ standardize แล้ว
    y_train, y_test  : numpy array ของ class_type (label)
    feature_names     : รายชื่อคอลัมน์ feature ทั้งหมด
    scaler            : StandardScaler ที่ fit แล้ว (เผื่อใช้ scale ข้อมูลใหม่ในอนาคต)
    """
    df = load_raw_data(csv_path)

    # animal_name เป็นแค่ identifier ไม่ใช่ feature ที่ใช้เทรน -> ตัดออก
    # class_type คือ label ที่เราต้องการทำนาย -> แยกออกมาต่างหาก
    feature_names = [c for c in df.columns if c not in ("animal_name", "class_type")]

    X = df[feature_names].values.astype(np.float32)
    y = df["class_type"].values.astype(np.int32)

    # แบ่งข้อมูลเป็น train 80% / test 20%
    # stratify=y เพื่อให้สัดส่วนแต่ละ class ใน train/test ใกล้เคียงกับข้อมูลเต็ม
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Standardize: ปรับให้ทุก feature มี mean=0, std=1
    # fit เฉพาะกับ train set แล้วนำค่าที่ได้ไป transform ทั้ง train และ test
    # (ห้าม fit กับ test set เพราะจะทำให้ข้อมูล "รั่ว" จาก test เข้ามาปนกับการเทรน)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    # ทดสอบว่าไฟล์นี้ทำงานถูกต้องเมื่อรันตรงๆ
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_prepare_data()

    print("Feature names:", feature_names)
    print("จำนวน features:", len(feature_names))
    print()
    print("ขนาด X_train:", X_train.shape)
    print("ขนาด X_test :", X_test.shape)
    print()
    print("ตัวอย่างข้อมูล X_train แถวแรก (หลัง standardize):")
    print(X_train[0])
    print()
    print("ตัวอย่าง label y_train 10 แถวแรก:", y_train[:10])
    print("ชื่อ class ที่ตรงกับ label:",
          [CLASS_NAMES[label] for label in y_train[:10]])