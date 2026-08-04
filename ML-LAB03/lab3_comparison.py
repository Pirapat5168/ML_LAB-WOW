import pandas as pd
import matplotlib.pyplot as plt

# ------------------ โหลดผลลัพธ์จาก LAB 1, LAB 2 ------------------
lab1 = pd.read_csv("data/lab1_results.csv")
lab2 = pd.read_csv("data/lab2_results.csv")

print("=" * 70)
print("หัวข้อที่ 1: Simple vs Multiple Linear Regression")
print("=" * 70)
print(lab1[["model", "test_mae", "test_rmse", "test_r2"]])
simple_r2 = lab1.loc[lab1["model"] == "Simple Linear Regression", "test_r2"].values[0]
multi_r2 = lab1.loc[lab1["model"] == "Multiple Linear Regression", "test_r2"].values[0]
diff_r2 = multi_r2 - simple_r2
print(f"\n-> Multiple LR มี Test R² สูงกว่า Simple LR อยู่ {diff_r2:.3f}")
print("-> แปลว่าการเพิ่มจำนวน features (จาก 1 เป็น 50 PCA components)")
print("   ช่วยให้โมเดลอธิบายความแปรปรวนของอายุได้ดีขึ้นชัดเจน")

# ================================================================
print("\n" + "=" * 70)
print("หัวข้อที่ 2: Training vs Testing Performance")
print("=" * 70)
lab1_gap = lab1.copy()
lab1_gap["r2_gap"] = lab1_gap["train_r2"] - lab1_gap["test_r2"]
lab1_gap["mae_gap"] = lab1_gap["test_mae"] - lab1_gap["train_mae"]
print(lab1_gap[["model", "train_r2", "test_r2", "r2_gap", "train_mae", "test_mae", "mae_gap"]])
print("\n-> ถ้า r2_gap (Train R² - Test R²) เป็นบวกมาก แปลว่าโมเดล overfit")
print("   คือเรียนรู้ training data ได้ดีเกินไป จนไม่ generalize ไปยังข้อมูลใหม่")

# ================================================================
print("\n" + "=" * 70)
print("หัวข้อที่ 3: Regression vs Classification")
print("=" * 70)
print("Regression (Age Prediction) ใช้ R², MAE, RMSE")
print(lab1[lab1["model"] == "Multiple Linear Regression"][["test_mae", "test_rmse", "test_r2"]])
print("\nClassification (Gender Prediction) ใช้ Accuracy, Precision, Recall, F1, AUC")
print(lab2[["accuracy", "precision", "recall", "f1_score", "auc"]])
print("\n-> ทั้งสองงานใช้ features ชุดเดียวกัน (PCA components จากภาพใบหน้า)")
print("   แต่ตัวชี้วัดเทียบกันตรงๆ ไม่ได้ เพราะเป็นคนละประเภทปัญหา:")
print("   - Regression ทำนายค่าต่อเนื่อง (อายุ) วัดด้วยค่าความคลาดเคลื่อน")
print("   - Classification ทำนายประเภท (เพศ) วัดด้วยความถูก/ผิดของ class")

# ================================================================
print("\n" + "=" * 70)
print("หัวข้อที่ 4: Model Performance Metrics (สรุปรวม)")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# กราฟซ้าย: Regression metrics
lab1.plot(x="model", y=["train_r2", "test_r2"], kind="bar", ax=axes[0], color=["skyblue", "salmon"])
axes[0].set_title("Regression: Train vs Test R²")
axes[0].set_ylabel("R² Score")
axes[0].set_xticklabels(lab1["model"], rotation=15, ha="right")

# กราฟขวา: Classification metrics
metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]
values = lab2[metrics].values[0]
axes[1].bar(metrics, values, color="mediumseagreen")
axes[1].set_title("Classification: Gender Prediction Metrics")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("data/lab3_comparison_summary.png")
print("บันทึกกราฟสรุป: data/lab3_comparison_summary.png")

# ------------------ บันทึกสรุปรวมเป็น CSV ------------------
summary = pd.DataFrame({
    "Task": ["Age Prediction (Simple LR)", "Age Prediction (Multiple LR)", "Gender Prediction (Logistic Reg.)"],
    "Type": ["Regression", "Regression", "Classification"],
    "Key_Metric_1": [f"R²={lab1.iloc[0]['test_r2']:.3f}", f"R²={lab1.iloc[1]['test_r2']:.3f}", f"Acc={lab2.iloc[0]['accuracy']:.3f}"],
    "Key_Metric_2": [f"MAE={lab1.iloc[0]['test_mae']:.2f}", f"MAE={lab1.iloc[1]['test_mae']:.2f}", f"F1={lab2.iloc[0]['f1_score']:.3f}"],
})
summary.to_csv("data/lab3_final_summary.csv", index=False)
print("\nบันทึกสรุปสุดท้าย: data/lab3_final_summary.csv")
print(summary)