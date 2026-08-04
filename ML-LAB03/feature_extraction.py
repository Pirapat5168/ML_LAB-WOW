import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ------------------ ตั้งค่า ------------------
IMG_DIR = "data/UTKFace_sampled"
LABELS_CSV = "data/labels.csv"
IMG_SIZE = (64, 64)      # resize ทุกภาพให้เท่ากัน
N_COMPONENTS = 50         # จำนวน PCA components ที่จะเก็บไว้

# ------------------ โหลด labels ------------------
df = pd.read_csv(LABELS_CSV)
print(f"จำนวนภาพทั้งหมดใน labels.csv: {len(df)}")

# ------------------ โหลดภาพ + resize + flatten ------------------
features = []
valid_idx = []

for i, fname in enumerate(df['filename']):
    fpath = os.path.join(IMG_DIR, fname)
    try:
        img = Image.open(fpath).convert("L")          # grayscale
        img = img.resize(IMG_SIZE)
        arr = np.array(img).flatten() / 255.0          # normalize 0-1
        features.append(arr)
        valid_idx.append(i)
    except Exception as e:
        print(f"ข้ามภาพ {fname}: {e}")

X_raw = np.array(features)
df = df.iloc[valid_idx].reset_index(drop=True)  # เก็บเฉพาะแถวที่โหลดภาพสำเร็จ

print(f"Feature matrix ก่อนทำ PCA: {X_raw.shape}")  # (จำนวนภาพ, 4096)

# ------------------ PCA ------------------
pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_raw)

explained = pca.explained_variance_ratio_.sum()
print(f"Feature matrix หลังทำ PCA: {X_pca.shape}")
print(f"PCA {N_COMPONENTS} components เก็บ variance ไว้ได้: {explained*100:.2f}%")

# ------------------ กราฟ explained variance ------------------
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.savefig("data/pca_explained_variance.png")
print("บันทึกกราฟ: data/pca_explained_variance.png")

# ------------------ บันทึกผลลัพธ์ ------------------
pca_cols = [f"pc_{i+1}" for i in range(N_COMPONENTS)]
df_pca = pd.DataFrame(X_pca, columns=pca_cols)
df_final = pd.concat([df[['filename', 'age', 'gender']], df_pca], axis=1)

df_final.to_csv("data/features_pca.csv", index=False)
print(f"บันทึก data/features_pca.csv เรียบร้อย: {df_final.shape}")