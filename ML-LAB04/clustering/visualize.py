import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ตั้งค่า font ให้รองรับภาษาไทยในกราฟ (เหมือนที่แก้ไว้ใน classification/evaluate.py)
plt.rcParams["font.family"] = "Tahoma"
plt.rcParams["axes.unicode_minus"] = False


def plot_elbow_curve(k_values, inertias, save_path):
    """
    Plot กราฟ Elbow Method เปรียบเทียบ Inertia ของแต่ละค่า k
    ใช้หาว่า k เท่าไหร่ถึงเหมาะสม (จุดที่กราฟเริ่มลาดลงช้าลง)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertias, marker="o", linewidth=2, markersize=7)

    ax.set_xlabel("k (จำนวนกลุ่ม)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method สำหรับหาค่า k ที่เหมาะสม")
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"บันทึก elbow curve ไว้ที่: {save_path}")


def plot_clusters_2d(X, cluster_labels, true_labels, class_names_map, save_path):
    """
    Plot กลุ่มข้อมูล (16 มิติ) โดยย่อเหลือ 2 มิติด้วย PCA ก่อนวาด
    ใช้สี = cluster ที่ K-Means จัดให้ และรูปทรงจุด = สายพันธุ์จริง
    เพื่อให้เห็นว่ากลุ่มที่ K-Means แบ่งตรงกับสายพันธุ์จริงแค่ไหน
    """
    # ย่อข้อมูลจาก 16 มิติ เหลือ 2 มิติ (แกน PC1, PC2) เพื่อวาดกราฟได้
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    explained_var = pca.explained_variance_ratio_
    n_clusters = len(np.unique(cluster_labels))

    fig, ax = plt.subplots(figsize=(9, 7))

    # วาดจุดแต่ละ cluster ด้วยสีต่างกัน
    cmap = plt.get_cmap("tab10")
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            color=cmap(cluster_id % 10),
            label=f"Cluster {cluster_id}",
            s=60, alpha=0.75, edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
    ax.set_title("การจัดกลุ่มด้วย K-Means (ย่อมิติด้วย PCA)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"บันทึก cluster scatter plot ไว้ที่: {save_path}")