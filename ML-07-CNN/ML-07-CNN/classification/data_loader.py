"""
data_loader.py
โหลด path รูปภาพทั้งหมดจากโฟลเดอร์ DATASET/<label>/ และข้ามไฟล์ที่เสีย/เปิดไม่ได้
"""
import cv2
from pathlib import Path


def get_classes(dataset_dir: str) -> list:
    """คืนรายชื่อ class (ชื่อโฟลเดอร์ย่อย) เรียงตามตัวอักษร"""
    dataset_dir = Path(dataset_dir)
    classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"ไม่พบโฟลเดอร์ class ใดๆ ใน {dataset_dir}")
    return classes


def load_image_paths(dataset_dir: str):
    """
    สแกนทุก class แล้วคืน (image_paths, labels, classes)
    - image_paths: list ของ path ไฟล์รูปที่เปิดได้จริง
    - labels: list ของ index class (int) ตรงตำแหน่งกับ image_paths
    - classes: list ชื่อ class เรียงตาม index
    ไฟล์ที่เสีย/เปิดไม่ได้ (corrupted) จะถูกข้ามและรายงานจำนวนที่ข้าม
    """
    classes = get_classes(dataset_dir)
    image_paths, labels = [], []
    skipped = 0

    for label_idx, class_name in enumerate(classes):
        class_dir = Path(dataset_dir) / class_name
        files = [f for f in class_dir.iterdir()
                 if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]

        for f in files:
            img = cv2.imread(str(f))  # None ถ้าไฟล์เสีย/เปิดไม่ได้
            if img is None:
                skipped += 1
                continue
            image_paths.append(str(f))
            labels.append(label_idx)

    print(f"[data_loader] โหลด path สำเร็จ {len(image_paths)} รูป "
          f"(ข้ามไฟล์เสีย {skipped} ไฟล์) จาก {len(classes)} classes: {classes}")

    return image_paths, labels, classes
