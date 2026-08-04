import os
import numpy as np

from data_loader import load_and_prepare_data, CLASS_NAMES
from knn_tf import KNNClassifierTF
from evaluate import (
    compute_accuracy,
    build_confusion_matrix,
    plot_confusion_matrix,
    plot_k_curve,
    save_predictions_csv,
)

# ค่า k ที่ต้องการทดลองเปรียบเทียบ ตามที่ใบงานกำหนด
K_VALUES = [3, 5, 7]

OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----- ขั้นตอนที่ 1: โหลดและเตรียมข้อมูล -----
    print("=" * 50)
    print("ขั้นตอนที่ 1: โหลดข้อมูล")
    print("=" * 50)
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_prepare_data()
    print(f"จำนวนข้อมูล train: {len(X_train)} ตัวอย่าง")
    print(f"จำนวนข้อมูล test : {len(X_test)} ตัวอย่าง")
    print(f"จำนวน features   : {len(feature_names)}")

    # เตรียม class labels ที่เรียงลำดับแล้ว (ใช้กับ confusion matrix)
    class_labels = sorted(CLASS_NAMES.keys())
    class_names_ordered = [CLASS_NAMES[c] for c in class_labels]

    # ----- ขั้นตอนที่ 2: เทรนและประเมินผลทุกค่า k -----
    print()
    print("=" * 50)
    print("ขั้นตอนที่ 2: เทรน KNN ด้วยค่า k ต่างๆ")
    print("=" * 50)

    results = {}  # k -> (accuracy, predictions)

    for k in K_VALUES:
        knn = KNNClassifierTF(k=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)

        accuracy = compute_accuracy(y_test, predictions)
        results[k] = (accuracy, predictions)

        print(f"k = {k}  ->  Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")

    # ----- ขั้นตอนที่ 3: หาค่า k ที่ดีที่สุด -----
    best_k = max(results, key=lambda k: results[k][0])
    best_accuracy, best_predictions = results[best_k]

    print()
    print("=" * 50)
    print(f"k ที่ดีที่สุดคือ k = {best_k} (Accuracy = {best_accuracy:.4f})")
    print("=" * 50)

    # ----- ขั้นตอนที่ 4: เซฟกราฟเปรียบเทียบ k -----
    k_values_list = list(results.keys())
    accuracy_list = [results[k][0] for k in k_values_list]
    plot_k_curve(
        k_values_list, accuracy_list,
        save_path=os.path.join(OUTPUT_DIR, "01_k_curve.png"),
    )

    # ----- ขั้นตอนที่ 5: สร้าง confusion matrix จากโมเดลที่ดีที่สุด -----
    conf_matrix = build_confusion_matrix(y_test, best_predictions, class_labels)
    plot_confusion_matrix(
        conf_matrix, class_names_ordered,
        save_path=os.path.join(OUTPUT_DIR, "02_confusion_matrix.png"),
    )

    # ----- ขั้นตอนที่ 6: เซฟผลการทำนายเป็น CSV -----
    save_predictions_csv(
        y_test, best_predictions, CLASS_NAMES,
        save_path=os.path.join(OUTPUT_DIR, "predictions.csv"),
    )

    # ----- ขั้นตอนที่ 7: สรุปผลการทดลอง (discussion) -----
    print()
    print("=" * 50)
    print("สรุปผลการทดลอง (Discussion)")
    print("=" * 50)
    for k in k_values_list:
        acc = results[k][0]
        n_correct = int(round(acc * len(y_test)))
        print(f"- k={k}: ทายถูก {n_correct}/{len(y_test)} ตัวอย่าง "
              f"(accuracy {acc*100:.2f}%)")

    print()
    print(f"สรุป: ค่า k ที่ดีที่สุดสำหรับ dataset นี้คือ k={best_k} "
          f"ให้ accuracy สูงสุดที่ {best_accuracy*100:.2f}%")
    print("เหตุผลที่เป็นไปได้: Zoo Dataset มี feature (เช่น hair, milk, feathers, "
          "fins) ที่แยกแต่ละ class ของสัตว์ได้ชัดเจนมาก ทำให้ระยะห่างระหว่าง "
          "จุดข้อมูลต่าง class มีค่าสูง และ k เพื่อนบ้านที่ใกล้ที่สุดมักจะเป็น "
          "class เดียวกันอยู่แล้ว จึงได้ accuracy สูงในทุกค่า k ที่ทดลอง")


if __name__ == "__main__":
    main()