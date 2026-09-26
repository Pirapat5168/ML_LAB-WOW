"""
split_data.py
แบ่งข้อมูลเป็น train / validation / test set (stratified ตาม label)
"""
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """
    แบ่งเป็น 3 ส่วน: train (70%) / val (15%) / test (15%) ตามค่า default
    ใช้ stratify=y เพื่อรักษาสัดส่วน class ในแต่ละ split ให้เท่ากัน
    """
    y = np.array(y)

    # แบ่ง test ออกก่อน
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # แบ่ง val จากส่วนที่เหลือ (ปรับสัดส่วนให้ val_size ตรงกับทั้งชุดข้อมูลจริง)
    val_ratio_of_temp = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp,
        stratify=y_temp, random_state=random_state
    )

    print(f"[split_data] train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
