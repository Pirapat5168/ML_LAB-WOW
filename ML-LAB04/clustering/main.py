"""
main.py
--------
สคริปต์หลักของ LAB 2: K-Means Clustering
รวมทุกขั้นตอน: โหลดข้อมูล -> หา k ที่เหมาะสมด้วย Elbow Method
-> เทรน K-Means จริง -> เปรียบเทียบกับสายพันธุ์จริง -> เซฟผลลัพธ์
"""

import os
import numpy as np
import pandas as pd

from data_loader import load_and_prepare_data, CLASS_NAMES
from kmeans_tf import KMeansTF
from visualize import plot_elbow_curve, plot_clusters_2d

OUTPUT_DIR = "outputs"

# ช่วงค่า k ที่จะทดลองสำหรับ Elbow Method
K_RANGE = list(range(1, 11))

# ค่า k สุดท้ายที่จะใช้เทรนจริง (เท่ากับจำนวนสายพันธุ์จริงใน Zoo Dataset
# เพื่อเปรียบเทียบว่า K-Means จัดกลุ่มได้ตรงกับสายพันธุ์จริงแค่ไหน)
FINAL_K = 7


def run_elbow_method(X_scaled):
    """รัน K-Means หลายค่า k แล้วเก็บ inertia ของแต่ละตัว สำหรับวาด elbow curve"""
    inertias = []
    for k in K_RANGE:
        kmeans = KMeansTF(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        print(f"k = {k:2d}  ->  Inertia = {kmeans.inertia_:.4f}")
    return inertias


def build_cluster_summary(cluster_labels, true_labels, n_clusters):
    """
    สร้างตารางสรุปแต่ละ cluster:
    - ขนาด cluster
    - สายพันธุ์จริงที่พบมากที่สุดใน cluster นั้น (dominant species)
    - purity = สัดส่วนของสมาชิกที่เป็นสายพันธุ์ dominant (ยิ่งใกล้ 1 ยิ่งดี
      แปลว่า cluster นี้ "บริสุทธิ์" มีแต่สัตว์สายพันธุ์เดียวกันอยู่)
    """
    rows = []
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_size = int(mask.sum())

        if cluster_size == 0:
            rows.append({
                "cluster_id": cluster_id,
                "size": 0,
                "dominant_species": "-",
                "purity": 0.0,
            })
            continue

        true_labels_in_cluster = true_labels[mask]
        values, counts = np.unique(true_labels_in_cluster, return_counts=True)
        dominant_label = values[np.argmax(counts)]
        dominant_count = counts.max()
        purity = dominant_count / cluster_size

        rows.append({
            "cluster_id": cluster_id,
            "size": cluster_size,
            "dominant_species": CLASS_NAMES[dominant_label],
            "purity": round(purity, 4),
        })

    return pd.DataFrame(rows)


def build_clustered_animals(animal_names, true_labels, cluster_labels):
    """สร้างตารางรายละเอียดทุกตัว: ชื่อสัตว์, สายพันธุ์จริง, cluster ที่ถูกจัดให้"""
    df = pd.DataFrame({
        "animal_name": animal_names,
        "true_species": [CLASS_NAMES[label] for label in true_labels],
        "assigned_cluster": cluster_labels,
    })
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----- ขั้นตอนที่ 1: โหลดข้อมูล -----
    print("=" * 50)
    print("ขั้นตอนที่ 1: โหลดข้อมูล")
    print("=" * 50)
    X_scaled, animal_names, true_labels, feature_names, scaler = load_and_prepare_data()
    print(f"จำนวนตัวอย่างทั้งหมด: {len(X_scaled)}")
    print(f"จำนวน features      : {len(feature_names)}")

    # ----- ขั้นตอนที่ 2: Elbow Method หาค่า k ที่เหมาะสม -----
    print()
    print("=" * 50)
    print("ขั้นตอนที่ 2: Elbow Method (ทดลอง k = 1 ถึง 10)")
    print("=" * 50)
    inertias = run_elbow_method(X_scaled)
    plot_elbow_curve(K_RANGE, inertias, save_path=os.path.join(OUTPUT_DIR, "01_elbow.png"))

    # ----- ขั้นตอนที่ 3: เทรน K-Means จริงด้วย FINAL_K -----
    print()
    print("=" * 50)
    print(f"ขั้นตอนที่ 3: เทรน K-Means จริงด้วย k = {FINAL_K}")
    print("=" * 50)
    kmeans = KMeansTF(n_clusters=FINAL_K, random_state=42)
    kmeans.fit(X_scaled)
    print(f"เทรนเสร็จภายใน {kmeans.n_iter_} รอบ, Inertia = {kmeans.inertia_:.4f}")

    # ----- ขั้นตอนที่ 4: วาด scatter plot ของกลุ่ม -----
    plot_clusters_2d(
        X_scaled, kmeans.labels_, true_labels, CLASS_NAMES,
        save_path=os.path.join(OUTPUT_DIR, "02_clusters.png"),
    )

    # ----- ขั้นตอนที่ 5: สร้างตารางสรุปผล และเซฟเป็น CSV -----
    print()
    print("=" * 50)
    print("ขั้นตอนที่ 5: สรุปผลและเปรียบเทียบกับสายพันธุ์จริง")
    print("=" * 50)

    summary_df = build_cluster_summary(kmeans.labels_, true_labels, FINAL_K)
    summary_path = os.path.join(OUTPUT_DIR, "cluster_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"บันทึกตารางสรุป cluster ไว้ที่: {summary_path}")
    print()
    print(summary_df.to_string(index=False))

    clustered_animals_df = build_clustered_animals(animal_names, true_labels, kmeans.labels_)
    clustered_path = os.path.join(OUTPUT_DIR, "clustered_animals.csv")
    clustered_animals_df.to_csv(clustered_path, index=False)
    print()
    print(f"บันทึกรายละเอียดสัตว์แต่ละตัวไว้ที่: {clustered_path}")

    # ----- ขั้นตอนที่ 6: สรุปผลการทดลอง (discussion) -----
    print()
    print("=" * 50)
    print("สรุปผลการทดลอง (Discussion)")
    print("=" * 50)
    avg_purity = summary_df["purity"].mean()
    print(f"- ใช้ k = {FINAL_K} กลุ่ม (เท่ากับจำนวนสายพันธุ์จริงในข้อมูล)")
    print(f"- Purity เฉลี่ยของทุก cluster = {avg_purity:.4f} "
          f"({avg_purity*100:.2f}%)")
    print("- Purity ที่สูง แปลว่าสมาชิกในแต่ละ cluster ส่วนใหญ่เป็นสายพันธุ์")
    print("  เดียวกันจริง แสดงว่า K-Means สามารถจับรูปแบบ (pattern) ของ")
    print("  feature ทางกายภาพ (มีขน, มีนม, มีปีก ฯลฯ) ได้สอดคล้องกับการจัด")
    print("  หมวดหมู่สายพันธุ์สัตว์จริงในทางชีววิทยาได้ดีในระดับหนึ่ง")
    print("- อย่างไรก็ตาม K-Means เป็น unsupervised learning จึงไม่รับประกัน")
    print("  ว่ากลุ่มที่ได้จะตรงกับ label จริงเป๊ะ 100% ต่างจาก classification")
    print("  ที่มี label กำกับตอนเทรนโดยตรง")


if __name__ == "__main__":
    main()