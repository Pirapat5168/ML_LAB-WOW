# ML Lab 2: Data Preprocessing

โปรเจกต์นี้เป็นส่วนหนึ่งของ **ใบงานที่ 2: Data Preprocessing**
วิชา Machine Learning (04-624-201)
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์
มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี (RMUTT)

## เกี่ยวกับ Dataset

**Loan Default Dataset** (Kaggle) — ข้อมูลการขอสินเชื่อบ้าน (mortgage loan) จากธนาคารในสหรัฐฯ
ใช้ทำนายว่าผู้กู้จะผิดนัดชำระหนี้ (default) หรือไม่ จากปัจจัยด้านรายได้ เครดิต และตัวเงินกู้

- แหล่งที่มา: https://www.kaggle.com/datasets/yasserh/loan-default-dataset
- จำนวนแถว: ~148,670 แถว
- จำนวนคอลัมน์: 34 คอลัมน์
- Target column: `Status` (0 = ไม่ default, 1 = default)

## โครงสร้างโปรเจกต์

```
ML-LAB02/
│
├── data/
│   └── Loan_Default.csv          # ข้อมูลดิบจาก Kaggle
│
├── notebook/
│   ├── lab1_dataset_exploration.ipynb
│   │   → โหลดข้อมูล, ดู shape/dtypes, summary statistics,
│   │     ตรวจ missing values, duplicate, class distribution
│   │
│   ├── lab2_data_visualization.ipynb
│   │   → Histogram ของทุก numeric column,
│   │     Correlation Heatmap ระหว่างฟีเจอร์
│   │
│   ├── part3_data_cleaning.ipynb
│   │   → จัดการ missing values (median/mode),
│   │     ลบ duplicate, แก้ incorrect data,
│   │     แปลง data type, เปรียบเทียบ mean vs median
│   │
│   └── part4_feature_engineering.ipynb
│       → Label Encoding (คอลัมน์ binary/ordinal),
│         One-Hot Encoding (คอลัมน์ nominal),
│         สร้าง dataset สุดท้ายพร้อมใช้ train model
│
├── venv/                          # Python virtual environment (ไม่ push ขึ้น GitHub)
├── requirements.txt               # รายชื่อ library ที่ใช้ทั้งหมด
├── .gitignore                     # กัน venv/ และไฟล์ไม่จำเป็นไม่ให้ขึ้น GitHub
└── README.md                      # ไฟล์นี้
```

## รายละเอียดแต่ละ Notebook

| Notebook | หัวข้อในใบงาน | Output หลัก |
|---|---|---|
| `lab1_dataset_exploration.ipynb` | LAB1: Dataset Exploration | ตาราง missing values, duplicate count, กราฟ class distribution |
| `lab2_data_visualization.ipynb` | LAB2: Data Visualization | Histogram 34 คอลัมน์, Correlation Heatmap |
| `part3_data_cleaning.ipynb` | Part 3: Data Cleaning | ข้อมูลที่สะอาดแล้ว (ไม่มี missing/duplicate), กราฟเทียบ mean-median |
| `part4_feature_engineering.ipynb` | Part 4: Feature Engineering | Dataset พร้อมใช้ train (encoded features) |

## วิธีติดตั้งและรันโปรเจกต์

```bash
# 1. Clone หรือดาวน์โหลดโปรเจกต์
cd ML-LAB02

# 2. สร้าง virtual environment
python -m venv venv

# 3. Activate venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 4. ติดตั้ง library ที่จำเป็น
pip install -r requirements.txt

# 5. เปิด VS Code แล้วรันไฟล์ .ipynb ใน notebook/
code .
```

## เทคนิคที่ใช้

- **Missing Value Handling:** เติมด้วย median (numeric) และ mode (categorical) เพื่อลดผลกระทบจาก outlier
- **Encoding Strategy:** เลือก Label Encoding สำหรับคอลัมน์ binary/ordinal และ One-Hot Encoding สำหรับคอลัมน์ nominal เพื่อไม่ให้โมเดลเข้าใจผิดว่ามีลำดับ

## ผู้จัดทำ

[พีรภัทร พุดพันธ์]
รหัสนักศึกษา: [116710400685-9]
กลุ่ม: [SEC 2]