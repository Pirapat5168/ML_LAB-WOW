# ใบงานที่ 6 — Neural Network และการประยุกต์ใช้งาน

**วิชา:** Machine Learning (04-624-201) | ภาควิชาวิศวกรรมคอมพิวเตอร์ | RMUTT
**หัวข้อ:** Neural Network (NN) on a Dataset of Your Choice — Gender Classification

---

## 1. Dataset ที่เลือกใช้

ใช้ dataset เดิมจาก **LAB 3** (age_gender / UTKFace) โดยต่อยอดจากงานเดิมที่ทำ
**PCA** บนภาพใบหน้าไว้แล้ว (`data/features_pca.csv`)

| รายละเอียด | ค่า |
|---|---|
| จำนวนข้อมูล | 1,500 ตัวอย่าง |
| Features | `pc_1 … pc_50` (50 principal components จากภาพใบหน้า) |
| Label (target) | `gender` → 0 = Male, 1 = Female |
| Class distribution | Male 818 / Female 682 |

**เหตุผลที่เลือก:** ใน LAB 3 เคยใช้ pipeline `StandardScaler → PCA → LogisticRegression`
ทำนาย gender ได้ accuracy ประมาณ 0.85 ใน LAB 6 นี้จึงนำ features ชุดเดียวกันมาลอง
แทนที่ Logistic Regression ด้วย **Neural Network** เพื่อเปรียบเทียบว่าโมเดลแบบ NN
จะให้ผลต่างจากโมเดลเชิงเส้นอย่างไร

---

## 2. โครงสร้างโปรเจกต์

```
mini-proj/
├── data/
│   └── features_pca.csv        # dataset (จาก LAB 3)
├── data_loader.py               # โหลด dataset
├── split_data.py                # แบ่ง train/test (stratify)
├── preprocessing.py             # StandardScaler
├── nn_model.py                  # สร้าง NN model + config ที่จะเปรียบเทียบ
├── evaluate.py                  # accuracy, confusion matrix, กราฟ
├── main.py                      # รันทั้ง pipeline (train + compare + save)
├── test_nn.py                   # โหลดโมเดลที่เทรนแล้วมาทดสอบกับข้อมูลใหม่
├── requirements.txt
└── outputs/                     # ผลลัพธ์ทั้งหมด (กราฟ, csv, โมเดล)
```

**วิธีรัน:**
```bash
pip install -r requirements.txt
python main.py        # เทรน + เปรียบเทียบ + บันทึกผลทั้งหมดลง outputs/
python test_nn.py     # ทดสอบโมเดลที่บันทึกไว้กับข้อมูลตัวอย่างใหม่
```

---

## 3. ขั้นตอนการทำงาน (ตามโจทย์)

1. **Select and load a dataset** → `data_loader.py`
2. **Split train/test** → `split_data.py` (80/20, stratify ตาม gender)
3. **Standardize input features** → `preprocessing.py` (StandardScaler fit บน train เท่านั้น)
4. **Build NN model** → `nn_model.py` (Dense layers + Dropout + sigmoid output, binary_crossentropy, Adam optimizer)
5. **Train ด้วย epoch ต่างกัน** → Experiment 1 ใน `main.py`
6. **เปรียบเทียบ NN configuration** (จำนวน hidden layer / neuron) → Experiment 2 ใน `main.py`
7. **Evaluate ด้วย accuracy** → `evaluate.py`

---

## 4. Neural Network Architecture

Input (50 features) → [Dense + Dropout 0.2] × N hidden layers → Dense(1, sigmoid)
Loss: `binary_crossentropy` | Optimizer: `Adam (lr=0.001)` | Metric: `accuracy`

### Experiment 1 — เปรียบเทียบจำนวน Epoch (architecture คงที่ = 64→32)

| Epochs | Accuracy |
|---|---|
| 10 | 0.823 |
| 30 | **0.843** |
| 60 | 0.807 |

> Epoch มากขึ้นไม่ได้แปลว่าดีขึ้นเสมอไป — ที่ 60 epoch เริ่มเห็นสัญญาณ overfitting
> (validation accuracy ในกราฟ `history_epoch60.png` เริ่มนิ่ง/ลดลง ขณะที่ train accuracy ยังสูงขึ้น)

### Experiment 2 — เปรียบเทียบ NN Configuration (epoch คงที่ = 30)

| Config | Hidden Layers | Accuracy |
|---|---|---|
| Small_1Layer | [16] | **0.837** |
| Medium_2Layer | [64, 32] | 0.810 |
| Large_3Layer | [128, 64, 32] | 0.820 |

> เนื่องจากข้อมูลมีเพียง 1,500 ตัวอย่างและ 50 features, โมเดลขนาดเล็ก (1 hidden layer)
> ให้ผลดีที่สุด สอดคล้องกับหลักการที่ว่าโมเดลใหญ่เกินไปเมื่อข้อมูลน้อยจะเสี่ยง overfitting

*(ตัวเลข accuracy อาจขยับเล็กน้อย ±1-2% ในแต่ละครั้งที่รัน เนื่องจาก random weight
initialization ของ Neural Network — ดูค่าจริงล่าสุดได้ที่ `outputs/results_summary.csv`)*

---

## 5. Output ที่ได้ (ใน `outputs/`)

- `history_<config>.png` — กราฟ training/validation accuracy & loss ต่อ epoch (7 ไฟล์)
- `confusion_matrix_<config>.png` — confusion matrix ของแต่ละ architecture
- `compare_epochs.png` — กราฟแท่งเปรียบเทียบ accuracy ตามจำนวน epoch
- `compare_architectures.png` — กราฟแท่งเปรียบเทียบ accuracy ตาม architecture
- `results_summary.csv` — ตารางสรุปผลทุก config
- `sample_predictions.csv` — ตัวอย่างการทำนาย 10 รายการจาก test set
- `best_model.keras` — โมเดลที่ accuracy สูงสุด (ใช้ต่อใน `test_nn.py`)
- `scaler_mean.npy`, `scaler_scale.npy` — ค่า StandardScaler ที่ fit ไว้ (ใช้กับข้อมูลใหม่)

---

## 6. สรุปผล

- Neural Network ให้ accuracy ในช่วง **0.81 – 0.84** สำหรับการจำแนก gender จาก
  PCA features ของภาพใบหน้า ใกล้เคียงกับ Logistic Regression ใน LAB 3
  (~0.85) เนื่องจาก dataset ผ่าน PCA มาแล้วทำให้ relationship ระหว่าง feature กับ
  label ค่อนข้างเป็นเส้นตรงอยู่แล้ว โมเดล NN จึงไม่ได้เหนือกว่ามากนัก
- Epoch ที่เหมาะสมอยู่ที่ประมาณ 30 (ไม่มากไม่น้อยเกินไป)
- Architecture เล็ก (1 hidden layer) เพียงพอกับขนาดข้อมูลชุดนี้ — architecture ใหญ่กว่า
  ไม่ได้ช่วยให้ผลดีขึ้น และเสี่ยง overfitting มากกว่า
