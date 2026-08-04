import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
import matplotlib.pyplot as plt

# ------------------ โหลดข้อมูล ------------------
df = pd.read_csv("data/features_pca.csv")
pc_cols = [c for c in df.columns if c.startswith("pc_")]

X = df[pc_cols].values
y = df["gender"].values   # 0 = male, 1 = female

print(f"จำนวนข้อมูลทั้งหมด: {len(df)}")
print(f"Gender distribution: {pd.Series(y).value_counts().to_dict()}")

# ------------------ แบ่ง train/test ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ================================================================
# 1) DECISION BOUNDARY VISUALIZATION (ใช้แค่ pc_1, pc_2 เพื่อ plot ได้ใน 2D)
# ================================================================
X_2d = df[["pc_1", "pc_2"]].values
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

model_2d = LogisticRegression(max_iter=1000)
model_2d.fit(X_train_2d, y_train_2d)

# สร้าง grid สำหรับวาด decision boundary
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm", edgecolors="k", alpha=0.7)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Decision Boundary (Logistic Regression, pc_1 vs pc_2)")
plt.legend(handles=scatter.legend_elements()[0], labels=["Male", "Female"])
plt.savefig("data/lab2_decision_boundary.png")
print("บันทึกกราฟ: data/lab2_decision_boundary.png")

# ================================================================
# 2) LOGISTIC REGRESSION (ใช้ทุก PCA components) - โมเดลหลัก
# ================================================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # ความน่าจะเป็นของ class 1 (female)

# ================================================================
# 3) GENDER PREDICTION - METRICS
# ================================================================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print()
print("=" * 60)
print(f"LOGISTIC REGRESSION (features: ทั้งหมด {len(pc_cols)} PCA components)")
print("=" * 60)
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")

# ================================================================
# 4) CONFUSION MATRIX
# ================================================================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix - Gender Prediction")
plt.savefig("data/lab2_confusion_matrix.png")
print("บันทึกกราฟ: data/lab2_confusion_matrix.png")

# ================================================================
# 5) ROC Curve + AUC
# ================================================================
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Gender Prediction")
plt.legend()
plt.savefig("data/lab2_roc_curve.png")
print(f"บันทึกกราฟ: data/lab2_roc_curve.png")
print(f"AUC: {roc_auc:.3f}")

# ================================================================
# บันทึกผลลัพธ์สรุปไว้ใช้ต่อใน LAB 3
# ================================================================
results = pd.DataFrame({
    "model": ["Logistic Regression (Gender)"],
    "accuracy": [acc],
    "precision": [prec],
    "recall": [rec],
    "f1_score": [f1],
    "auc": [roc_auc],
})
results.to_csv("data/lab2_results.csv", index=False)
print("\nบันทึกสรุปผล: data/lab2_results.csv")
print(results)