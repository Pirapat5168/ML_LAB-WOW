import tensorflow as tf
import numpy as np


class KNNClassifierTF:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """เก็บข้อมูล train ไว้ใช้ตอน predict (ไม่มีการคำนวณอะไรตอนนี้)"""
        self.X_train = tf.constant(X_train, dtype=tf.float32)
        self.y_train = tf.constant(y_train, dtype=tf.int32)

    def _compute_distances(self, X_test):
         X_test = tf.constant(X_test, dtype=tf.float32)
         diff = tf.expand_dims(X_test, axis=1) - tf.expand_dims(self.X_train, axis=0)

         squared_diff = tf.square(diff)
         sum_squared = tf.reduce_sum(squared_diff, axis=2)
         distances = tf.sqrt(sum_squared)


         return distances
    def predict(self, X_test):
        """
        ทำนาย class ของแต่ละจุดใน X_test

        คืนค่าเป็น numpy array ของ label ที่ทำนายได้ (ความยาว = n_test)
        """
        distances = self._compute_distances(X_test)  # (n_test, n_train)

        # หา index ของ k จุดที่ใกล้ที่สุด (ระยะน้อยที่สุด) สำหรับแต่ละแถว
        # tf.nn.top_k หาค่ามากที่สุด เราจึงใส่ -distances เพื่อกลับเป็นหาค่าน้อยที่สุดแทน
        _, nearest_indices = tf.nn.top_k(-distances, k=self.k)  # (n_test, k)

        # ดึง label ของ k เพื่อนบ้านที่ใกล้ที่สุดออกมา
        nearest_labels = tf.gather(self.y_train, nearest_indices)  # (n_test, k)

        predictions = []
        for row in nearest_labels.numpy():
            # majority vote: หา label ที่ปรากฏบ่อยที่สุดในบรรดา k เพื่อนบ้าน
            values, counts = np.unique(row, return_counts=True)
            majority_label = values[np.argmax(counts)]
            predictions.append(majority_label)

        return np.array(predictions)


if __name__ == "__main__":
    # ทดสอบด้วยข้อมูลจริงจาก data_loader
    from data_loader import load_and_prepare_data, CLASS_NAMES

    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_prepare_data()

    knn = KNNClassifierTF(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    print("Label จริง   :", y_test)
    print("Label ที่ทาย :", predictions)

    accuracy = np.mean(predictions == y_test)
    print(f"\nAccuracy (k=3): {accuracy:.4f} ({accuracy*100:.2f}%)")