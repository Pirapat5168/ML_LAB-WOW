"""
preprocessing.py
resize รูปภาพให้ขนาดเท่ากันทั้งหมด และแปลงสี BGR (cv2 อ่านมาเป็น BGR) -> RGB
"""
import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE = (128, 128)  # (width, height) - ปรับได้ตามสเปกเครื่อง/เวลาที่มี


def load_and_preprocess(image_paths, img_size=IMG_SIZE):
    """
    โหลดรูปจาก path, resize, แปลง BGR->RGB, normalize พิกเซลเป็น [0,1]
    คืน:
      - features: numpy array shape (N, H, W, 3) dtype float32
      - valid_idx: index ของ image_paths ที่โหลดสำเร็จ (ใช้จับคู่กับ labels เดิม)
    """
    features = []
    valid_idx = []

    for i, path in enumerate(tqdm(image_paths, desc="Preprocessing")):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        features.append(img)
        valid_idx.append(i)

    return np.array(features, dtype=np.float32), valid_idx
