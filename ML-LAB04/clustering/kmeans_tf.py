import numpy as np
import tensorflow as tf

from knn_tools import assign_nearest_centroid, compute_inertia


class KMeansTF:
    """
    K-Means Clustering ที่ implement ด้วย TensorFlow

    อัลกอริทึม (Lloyd's algorithm):
    1. สุ่มเลือกจุดข้อมูล k จุดมาเป็น centroid เริ่มต้น
    2. Assign: ให้แต่ละจุดเข้ากลุ่ม centroid ที่ใกล้ที่สุด (1-NN)
    3. Update: คำนวณ centroid ใหม่จากค่าเฉลี่ยของจุดในแต่ละกลุ่ม
    4. ทำซ้ำขั้นตอน 2-3 จนกว่า centroid จะหยุดขยับ (converge)
       หรือครบจำนวนรอบสูงสุด
    """

    def __init__(self, n_clusters=7, max_iters=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _init_centroids(self, X):
        """สุ่มเลือกจุดข้อมูลจริง k จุดมาเป็น centroid เริ่มต้น"""
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        return tf.constant(X[indices], dtype=tf.float32)

    def fit(self, X):
        """เทรน K-Means บนข้อมูล X"""
        X_tf = tf.constant(X, dtype=tf.float32)
        centroids = self._init_centroids(X)

        rng = np.random.RandomState(self.random_state)

        for iteration in range(self.max_iters):
            # ขั้นตอน Assign: หา centroid ที่ใกล้ที่สุดของแต่ละจุด
            cluster_assignments, _ = assign_nearest_centroid(X_tf, centroids)
            assignments_np = cluster_assignments.numpy()

            # ขั้นตอน Update: คำนวณ centroid ใหม่จากค่าเฉลี่ยของแต่ละกลุ่ม
            new_centroids = np.zeros_like(centroids.numpy())
            for cluster_id in range(self.n_clusters):
                points_in_cluster = X[assignments_np == cluster_id]

                if len(points_in_cluster) == 0:
                    # กรณีไม่มีจุดไหนเข้ากลุ่มนี้เลย (cluster ว่าง)
                    # แก้ปัญหาด้วยการสุ่มจุดข้อมูลใหม่มาแทน centroid นี้
                    random_idx = rng.randint(0, X.shape[0])
                    new_centroids[cluster_id] = X[random_idx]
                else:
                    new_centroids[cluster_id] = points_in_cluster.mean(axis=0)

            new_centroids_tf = tf.constant(new_centroids, dtype=tf.float32)

            # เช็คว่า centroid ขยับน้อยกว่า tolerance หรือยัง (converge แล้ว)
            shift = tf.reduce_sum(tf.square(new_centroids_tf - centroids))
            centroids = new_centroids_tf
            self.n_iter_ = iteration + 1

            if shift.numpy() < self.tol:
                break

        # คำนวณผลลัพธ์สุดท้ายหลังจบการเทรน
        final_assignments, _ = assign_nearest_centroid(X_tf, centroids)
        self.labels_ = final_assignments.numpy()
        self.centroids_ = centroids.numpy()
        self.inertia_ = compute_inertia(X_tf, centroids, final_assignments)

        return self

    def predict(self, X):
        """ทำนายว่าจุดข้อมูลใหม่ควรอยู่กลุ่มไหน (โดยใช้ centroid ที่เทรนไว้แล้ว)"""
        X_tf = tf.constant(X, dtype=tf.float32)
        assignments, _ = assign_nearest_centroid(X_tf, self.centroids_)
        return assignments.numpy()


if __name__ == "__main__":
    from data_loader import load_and_prepare_data, CLASS_NAMES

    X_scaled, animal_names, true_labels, feature_names, scaler = load_and_prepare_data()

    # ทดลองด้วย k=7 (เท่ากับจำนวนสายพันธุ์จริงใน Zoo Dataset) เพื่อดูว่า
    # K-Means จะจัดกลุ่มได้ใกล้เคียงสายพันธุ์จริงแค่ไหน
    kmeans = KMeansTF(n_clusters=7, random_state=42)
    kmeans.fit(X_scaled)

    print(f"เทรนเสร็จภายใน {kmeans.n_iter_} รอบ")
    print(f"Inertia (WCSS): {kmeans.inertia_:.4f}")
    print()

    print("จำนวนสมาชิกในแต่ละกลุ่ม (cluster):")
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} ตัว")