"""
split_data.py
---------------
แบ่งข้อมูลเป็น training set และ testing set
"""

from sklearn.model_selection import train_test_split


def split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    แบ่งข้อมูลแบบ stratify ตาม y (gender) เพื่อให้สัดส่วน class
    ใน train/test ใกล้เคียงกับข้อมูลทั้งหมด
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
