import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ------------------ โหลดข้อมูล ------------------
df = pd.read_csv("data/features_pca.csv")
pc_cols = [c for c in df.columns if c.startswith("pc_")]

y = df["age"].values

# ------------------ แบ่ง train/test ------------------
# ใช้ index เดียวกันสำหรับทั้ง simple และ multiple เพื่อเทียบกันแฟร์ๆ
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42
)

# ================================================================
# 1) SIMPLE LINEAR REGRESSION (ใช้แค่ 1 feature: pc_1)
# ================================================================
X_simple = df[["pc_1"]].values

X_train_s, X_test_s = X_simple[train_idx], X_simple[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train)

pred_train_s = model_simple.predict(X_train_s)
pred_test_s = model_simple.predict(X_test_s)

print("=" * 60)
print("SIMPLE LINEAR REGRESSION (feature: pc_1 เท่านั้น)")
print("=" * 60)
print(f"Train  -> MAE: {mean_absolute_error(y_train, pred_train_s):.2f}, "
      f"RMSE: {np.sqrt(mean_squared_error(y_train, pred_train_s)):.2f}, "
      f"R2: {r2_score(y_train, pred_train_s):.3f}")
print(f"Test   -> MAE: {mean_absolute_error(y_test, pred_test_s):.2f}, "
      f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_test_s)):.2f}, "
      f"R2: {r2_score(y_test, pred_test_s):.3f}")

# ================================================================
# 2) MULTIPLE LINEAR REGRESSION (ใช้ทุก PCA components)
# ================================================================
X_multi = df[pc_cols].values

X_train_m, X_test_m = X_multi[train_idx], X_multi[test_idx]

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train)

pred_train_m = model_multi.predict(X_train_m)
pred_test_m = model_multi.predict(X_test_m)

print()
print("=" * 60)
print(f"MULTIPLE LINEAR REGRESSION (features: ทั้งหมด {len(pc_cols)} PCA components)")
print("=" * 60)
print(f"Train  -> MAE: {mean_absolute_error(y_train, pred_train_m):.2f}, "
      f"RMSE: {np.sqrt(mean_squared_error(y_train, pred_train_m)):.2f}, "
      f"R2: {r2_score(y_train, pred_train_m):.3f}")
print(f"Test   -> MAE: {mean_absolute_error(y_test, pred_test_m):.2f}, "
      f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_test_m)):.2f}, "
      f"R2: {r2_score(y_test, pred_test_m):.3f}")

# ================================================================
# 3) กราฟเปรียบเทียบ: Actual vs Predicted Age
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, pred_test_s, alpha=0.5, color="orange")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel("Actual Age")
axes[0].set_ylabel("Predicted Age")
axes[0].set_title(f"Simple LR (R2={r2_score(y_test, pred_test_s):.3f})")

axes[1].scatter(y_test, pred_test_m, alpha=0.5, color="teal")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_xlabel("Actual Age")
axes[1].set_ylabel("Predicted Age")
axes[1].set_title(f"Multiple LR (R2={r2_score(y_test, pred_test_m):.3f})")

plt.tight_layout()
plt.savefig("data/lab1_regression_comparison.png")
print("\nบันทึกกราฟ: data/lab1_regression_comparison.png")

# ================================================================
# บันทึกผลลัพธ์สรุปไว้ใช้ต่อใน LAB 3
# ================================================================
results = pd.DataFrame({
    "model": ["Simple Linear Regression", "Multiple Linear Regression"],
    "train_mae": [mean_absolute_error(y_train, pred_train_s), mean_absolute_error(y_train, pred_train_m)],
    "test_mae": [mean_absolute_error(y_test, pred_test_s), mean_absolute_error(y_test, pred_test_m)],
    "train_rmse": [np.sqrt(mean_squared_error(y_train, pred_train_s)), np.sqrt(mean_squared_error(y_train, pred_train_m))],
    "test_rmse": [np.sqrt(mean_squared_error(y_test, pred_test_s)), np.sqrt(mean_squared_error(y_test, pred_test_m))],
    "train_r2": [r2_score(y_train, pred_train_s), r2_score(y_train, pred_train_m)],
    "test_r2": [r2_score(y_test, pred_test_s), r2_score(y_test, pred_test_m)],
})
results.to_csv("data/lab1_results.csv", index=False)
print("บันทึกสรุปผล: data/lab1_results.csv")
print(results)