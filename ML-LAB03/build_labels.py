import os
import pandas as pd

target_dir = "data/UTKFace_sampled"

records = []
skipped = 0

for fname in os.listdir(target_dir):
    if not fname.endswith(".jpg"):
        continue
    try:
        age, gender, race, _ = fname.split("_", 3)
        records.append({
            "filename": fname,
            "age": int(age),
            "gender": int(gender),   # 0 = male, 1 = female
            "race": int(race)
        })
    except ValueError:
        skipped += 1
        continue

df = pd.DataFrame(records)
print(f"Parse สำเร็จ: {len(df)} แถว, ข้ามไป: {skipped} ไฟล์")
print(df.describe())
print(df['gender'].value_counts())

df.to_csv("data/labels.csv", index=False)
print("บันทึก data/labels.csv เรียบร้อย")