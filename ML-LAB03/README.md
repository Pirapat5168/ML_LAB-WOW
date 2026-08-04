# Age & Gender Prediction from Facial Images

โปรเจคสำหรับใบงานที่ 3 (Regression & Classification) วิชา Machine Learning
มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี (RMUTT)

ทำนาย**อายุ** (Regression) และ**เพศ** (Classification) จากภาพใบหน้า โดยใช้ UTKFace Dataset

---

## Overview

| หัวข้อ | รายละเอียด |
|---|---|
| Dataset | [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) (สุ่มมา 1,500 ภาพ จากทั้งหมด 20,000+ ภาพ) |
| Feature Engineering | Grayscale → Resize 64x64 → Flatten (4,096 features) → PCA (50 components, เก็บ variance ~89%) |
| Regression Task | ทำนายอายุ ด้วย Simple & Multiple Linear Regression |
| Classification Task | ทำนายเพศ ด้วย Logistic Regression |

## Results Summary

| Task | Type | Metric 1 | Metric 2 |
|---|---|---|---|
| Age Prediction (Simple LR) | Regression | R² ≈ -0.03 | MAE ≈ 15.5 |
| Age Prediction (Multiple LR) | Regression | R² ≈ 0.24 | MAE ≈ 12.7 |
| Gender Prediction (Logistic Regression) | Classification | Accuracy ≈ 0.84 | F1 ≈ 0.83 |

**สรุป:** Gender Classification ทำได้แม่นยำกว่า Age Regression อย่างชัดเจน เพราะเพศเป็นลักษณะที่แยกกลุ่มได้ชัดเจนกว่าค่าอายุที่ต่อเนื่องและมีปัจจัยกวนเยอะ ดูรายละเอียดการวิเคราะห์เต็มได้ใน notebook

