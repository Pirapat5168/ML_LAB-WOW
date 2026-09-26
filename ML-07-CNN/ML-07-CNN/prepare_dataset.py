"""
prepare_dataset.py
สคริปต์ช่วยเตรียม DATASET/ จาก Kaggle "140k Real and Fake Faces" (xhlulu)
สุ่มเลือกรูปจำนวนที่กำหนดจากแต่ละ split (train/valid/test) ของแต่ละ class
แล้ว copy รวมเป็น DATASET/real/ และ DATASET/fake/ ตามโครงสร้างที่ Lab ต้องการ
(ไม่ต้องมี train/valid/test ซ้อนอยู่ข้างใน เพราะ split_data.py จะ split เองอีกที)

วิธีใช้ (รันจากโฟลเดอร์ ML-07-CNN/):
    python prepare_dataset.py --source <path_to_real_vs_fake_folder> --n_per_class 3000

<path_to_real_vs_fake_folder> คือโฟลเดอร์ที่แตกไฟล์ zip จาก Kaggle มา
ซึ่งข้างในต้องมี train/, valid/, test/ (แต่ละอันมี real/ กับ fake/ ย่อย)
"""
import argparse
import random
import shutil
from pathlib import Path


def collect_all_images(source_dir: Path, class_name: str):
    """รวม path รูปทั้งหมดของ class นี้จากทุก split (train/valid/test)"""
    paths = []
    for split in ["train", "valid", "test"]:
        class_dir = source_dir / split / class_name
        if class_dir.exists():
            paths.extend(class_dir.glob("*.jpg"))
            paths.extend(class_dir.glob("*.png"))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                         help="path ไปยังโฟลเดอร์ real_vs_fake/real-vs-fake/ ที่มี train/valid/test อยู่ข้างใน")
    parser.add_argument("--dest", default="DATASET",
                         help="โฟลเดอร์ปลายทาง (default: DATASET)")
    parser.add_argument("--n_per_class", type=int, default=3000,
                         help="จำนวนรูปที่จะสุ่มต่อ class (default: 3000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    source_dir = Path(args.source)
    dest_dir = Path(args.dest)

    for class_name in ["real", "fake"]:
        all_paths = collect_all_images(source_dir, class_name)
        print(f"[prepare_dataset] {class_name}: พบทั้งหมด {len(all_paths)} รูป")

        if len(all_paths) < args.n_per_class:
            print(f"  คำเตือน: มีรูปน้อยกว่าที่ขอ ใช้ทั้งหมด {len(all_paths)} รูปแทน")
            selected = all_paths
        else:
            selected = random.sample(all_paths, args.n_per_class)

        out_dir = dest_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, src_path in enumerate(selected):
            dst_path = out_dir / f"{class_name}_{i:05d}{src_path.suffix}"
            shutil.copy2(src_path, dst_path)

        print(f"  copy แล้ว {len(selected)} รูป -> {out_dir}")

    print("\n[prepare_dataset] เสร็จสิ้น! ตรวจสอบโฟลเดอร์ DATASET/ ได้เลย")


if __name__ == "__main__":
    main()
