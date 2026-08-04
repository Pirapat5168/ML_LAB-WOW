import os
import random
import shutil
import pandas as pd

source_dir = "data/data/UTKFace_raw/UTKFace"
target_dir = "data/UTKFace_sampled"

os.makedirs(target_dir, exist_ok=True)

all_files = [f for f in os.listdir(source_dir) if f.endswith(".jpg")]
print(f"ภาพทั้งหมด: {len(all_files)}")

random.seed(42)
N_SAMPLES = 1500
sampled_files = random.sample(all_files, N_SAMPLES)

for fname in sampled_files:
    shutil.copy(os.path.join(source_dir, fname), os.path.join(target_dir, fname))

print(f"copy เสร็จแล้ว: {len(os.listdir(target_dir))} ไฟล์")