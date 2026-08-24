"""
preprocessing.py
------------------
Standardize input features ก่อนนำเข้า Neural Network
(ข้อมูล pc_1..pc_50 ผ่าน PCA มาจาก LAB 3 แล้ว แต่ยังต้อง
StandardScaler อีกครั้งบน training set เพื่อให้ทุก feature
มี mean=0, std=1 ตามที่โจทย์ LAB 6 กำหนด: "Standardize the
input features before training.")
"""

from sklearn.preprocessing import StandardScaler


def standardize(X_train, X_test):
    """
    fit StandardScaler บน X_train เท่านั้น แล้ว transform ทั้ง train/test
    (ป้องกัน data leakage จาก test set)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
