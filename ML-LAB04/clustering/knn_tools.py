import tensorflow as tf


def compute_distances(X, centroids):
    """
    คำนวณ Euclidean distance จากทุกจุดใน X ไปยังทุก centroid
    แบบ vectorized (หลักการเดียวกับ knn_tf.py)

    Parameters
    ----------
    X         : tensor รูปร่าง (n_points, n_features)
    centroids : tensor รูปร่าง (n_clusters, n_features)

    Returns
    -------
    distances : tensor รูปร่าง (n_points, n_clusters)
    """
    X = tf.cast(X, tf.float32)
    centroids = tf.cast(centroids, tf.float32)

    # ขยายมิติให้ broadcast กันได้ เหมือนใน knn_tf.py:
    # X: (n_points, 1, n_features)
    # centroids: (1, n_clusters, n_features)
    diff = tf.expand_dims(X, axis=1) - tf.expand_dims(centroids, axis=0)
    squared_diff = tf.square(diff)
    sum_squared = tf.reduce_sum(squared_diff, axis=2)
    distances = tf.sqrt(sum_squared)

    return distances  # shape: (n_points, n_clusters)


def assign_nearest_centroid(X, centroids):
    """
    หาว่าแต่ละจุดใน X ควรอยู่กลุ่มไหน โดยเลือก centroid ที่ใกล้ที่สุด
    (คือการทำ 1-Nearest Neighbor ระหว่างจุดข้อมูลกับกลุ่ม centroid)

    Returns
    -------
    cluster_assignments : tensor รูปร่าง (n_points,) ค่า index ของ centroid
                           ที่ใกล้ที่สุดของแต่ละจุด
    distances           : matrix ระยะห่างทั้งหมด (เผื่อใช้คำนวณ inertia ต่อ)
    """
    distances = compute_distances(X, centroids)

    # หา centroid ที่ใกล้ที่สุด (ระยะน้อยที่สุด) ของแต่ละจุด
    cluster_assignments = tf.argmin(distances, axis=1)

    return cluster_assignments, distances


def compute_inertia(X, centroids, cluster_assignments):
    """
    คำนวณ Inertia (Within-Cluster Sum of Squares - WCSS)
    ค่านี้ยิ่งน้อยยิ่งดี แปลว่าจุดข้อมูลอยู่ใกล้ centroid ของตัวเองมาก
    ใช้สำหรับวาด Elbow curve หาค่า k ที่เหมาะสม
    """
    X = tf.cast(X, tf.float32)
    assigned_centroids = tf.gather(centroids, cluster_assignments)
    squared_dist = tf.reduce_sum(tf.square(X - assigned_centroids), axis=1)
    inertia = tf.reduce_sum(squared_dist)
    return float(inertia.numpy())



