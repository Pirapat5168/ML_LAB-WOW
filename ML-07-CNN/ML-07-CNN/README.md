# ใบงานที่ 7 — Convolutional Neural Network และการประยุกต์ใช้งาน
**รายวิชา** Machine Learning (04-624-201) | **หัวข้อ:** CNN on a Dataset of Your Choice
**ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี**

---

## 1. วัตถุประสงค์

ตามที่ระบุในใบงาน:
1. นำ CNN มาประยุกต์ใช้จำแนกข้อมูลภาพที่เลือกเอง (dataset of choice)
2. เปรียบเทียบประสิทธิภาพของ CNN เมื่อปรับจำนวน epoch และโครงสร้าง (configuration) ที่แตกต่างกัน เช่น จำนวน convolutional layer และจำนวน neuron

## 2. Dataset ที่เลือกใช้

**140k Real and Fake Faces** (โดย xhlulu, เผยแพร่บน Kaggle)

| รายละเอียด | ค่า |
|---|---|
| ประเภทงาน | Binary classification (real vs fake) |
| จำนวน class | 2 class: `real` (ภาพใบหน้าจริงจาก Flickr/FFHQ), `fake` (ภาพใบหน้าที่สร้างจาก StyleGAN) |
| จำนวนรูปทั้งหมดในชุดต้นฉบับ | 140,000 รูป (70,000 รูป/class) |
| จำนวนรูปที่ใช้จริงในการทดลองนี้ | 12,000 รูป (6,000 รูป/class) สุ่มเลือกด้วย `prepare_dataset.py` |
| เหตุผลที่เลือก dataset นี้ | เป็นงาน binary classification ที่ท้าทายกว่าชุดข้อมูล benchmark ทั่วไป (เช่น cats vs dogs) เนื่องจากความแตกต่างระหว่างภาพจริงและภาพที่สร้างจาก GAN นั้นละเอียดอ่อนและใกล้เคียงกันมาก ทำให้เหมาะสำหรับทดสอบความสามารถของ CNN ในการจับ pattern ที่ซับซ้อน |

## 3. โครงสร้างโปรเจกต์และ Pipeline

```
ML-07-CNN/
├── DATASET/                    # real/, fake/ (เตรียมด้วย prepare_dataset.py)
├── prepare_dataset.py          # สุ่มเลือกรูปจาก Kaggle มาจัดเป็น DATASET/
├── classification/
│   ├── data_loader.py          # โหลด path รูป + ข้ามไฟล์เสีย
│   ├── preprocessing.py        # resize 128x128 + แปลง BGR→RGB + normalize
│   ├── split_data.py           # แบ่ง train/val/test แบบ stratified (70/15/15)
│   ├── cnn_model.py            # สร้าง/เทรน/บันทึก/ทำนายด้วย CNN (ปรับ config ได้)
│   ├── evaluate.py             # accuracy, classification report, confusion matrix, กราฟเปรียบเทียบ
│   ├── test_cnn.py             # ทดสอบโมเดลกับภาพสุ่ม 4 รูป
│   ├── main.py                 # pipeline หลัก: เทรนและเปรียบเทียบ 4 configuration
│   └── outputs/                # ผลลัพธ์ทั้งหมด (โมเดล, กราฟ, ค่า accuracy)
└── requirements.txt
```

**ขั้นตอนการทำงาน (data flow):**
โหลดรูป (`data_loader.py`) → resize + normalize (`preprocessing.py`) → แบ่ง train/val/test (`split_data.py`) → เทรน CNN 4 configuration เปรียบเทียบกัน (`cnn_model.py`) → ประเมินผลและสร้างกราฟ (`evaluate.py`) → ทดสอบกับภาพสุ่ม (`test_cnn.py`)

## 4. โครงสร้างโมเดล CNN

โมเดลถูกออกแบบให้ปรับ configuration ได้ผ่านพารามิเตอร์ ประกอบด้วย:

```
Input (128×128×3)
  → [Conv2D → BatchNorm → MaxPooling2D] × N   (N = จำนวน conv layer ที่ปรับได้)
  → Flatten
  → Dense(neurons, ReLU)                       (จำนวน neuron ที่ปรับได้)
  → Dropout(0.3)
  → Dense(1, Sigmoid)                          (binary classification)
```

- **Optimizer:** Adam, **Loss:** Binary Crossentropy
- **EarlyStopping:** หยุดเทรนอัตโนมัติเมื่อ `val_accuracy` ไม่ดีขึ้นติดต่อกัน 3 epoch พร้อมดึง weight ที่ดีที่สุดกลับมาใช้ (`restore_best_weights=True`) เพื่อป้องกัน overfitting และลดเวลาเทรนที่ไม่จำเป็น

## 5. การออกแบบการทดลอง (Configuration Comparison)

ทดลองเปรียบเทียบ CNN ทั้งหมด 4 configuration โดยปรับ 3 ตัวแปร: จำนวน convolutional layer, จำนวน neuron ใน dense layer, และจำนวน epoch สูงสุด

| Config | Conv Layers | Dense Neurons | Max Epochs | **Best Val. Accuracy** |
|---|:---:|:---:|:---:|:---:|
| cfg1_conv2_dense64_ep10 | 2 | 64 | 10 | 79.96% |
| cfg2_conv3_dense128_ep10 | 3 | 128 | 10 | 74.07% |
| **cfg3_conv3_dense128_ep20** | **3** | **128** | **20** | **🏆 86.79%** |
| cfg4_conv4_dense256_ep20 | 4 | 256 | 20 | 85.01% |

**Config ที่ดีที่สุด:** `cfg3_conv3_dense128_ep20` ถูกเลือกเป็นโมเดลสุดท้าย (บันทึกเป็น `outputs/cnn_model.keras`)

## 6. ผลการทดลองบนชุดทดสอบ (Test Set)

| Metric | ค่า |
|---|---|
| **Test Accuracy** | 86.83% |
| Precision (fake) | 0.8877 |
| Recall (fake) | 0.8433 |
| Precision (real) | 0.8508 |
| Recall (real) | 0.8933 |
| F1-score เฉลี่ย (macro avg) | 0.8683 |

ดูรายละเอียดเพิ่มเติมได้ที่:
- `outputs/confusion_matrix.png` — confusion matrix ของโมเดลที่ดีที่สุด
- `outputs/comparison_training_performance.png` — กราฟเปรียบเทียบ training/validation accuracy และ loss ของทั้ง 4 config
- `outputs/training_history.png` — กราฟ accuracy/loss ของโมเดลที่ดีที่สุดเพียงตัวเดียว
- `outputs/prediction_sample.png` — ตัวอย่างผลทำนายจากภาพสุ่ม 4 รูป

## 7. การวิเคราะห์และอภิปรายผล (Discussion)

**7.1 จำนวน epoch ที่มากขึ้นไม่ได้แปลว่า accuracy จะดีขึ้นเสมอไป**
เมื่อเปรียบเทียบ `cfg2` (epoch=10) กับโครงสร้างเดียวกันที่ `cfg3` (epoch=20) พบว่า `cfg3` ให้ผลดีกว่าอย่างชัดเจน (86.79% vs 74.07%) แสดงว่าโมเดลยังไม่ converge เต็มที่ที่ epoch 10 อย่างไรก็ตาม เมื่อเทียบ `cfg3` กับ `cfg4` ที่เพิ่มความซับซ้อนของโมเดล (conv layer 4 ชั้น, neuron 256) แต่ epoch เท่ากัน กลับได้ผลด้อยกว่าเล็กน้อย (85.01% vs 86.79%) สะท้อนว่าการเพิ่มความลึกของโมเดลมากเกินไปไม่ได้ช่วยให้ผลดีขึ้นเสมอไป และอาจทำให้เทรนไม่เสถียรมากขึ้น

**7.2 ความไม่เสถียรของ Validation Loss ในช่วงเริ่มต้นการเทรน (Loss Spike)**
จากกราฟ `comparison_training_performance.png` พบว่า `cfg1` (conv layer น้อยที่สุด) มีค่า validation loss พุ่งสูงผิดปกติในช่วง epoch แรก (สูงถึง ~17) ก่อนจะลดลงอย่างรวดเร็ว ปรากฏการณ์นี้เรียกว่า **loss spike** ซึ่งพบได้บ่อยในโมเดลที่มีความจุ (capacity) ต่ำเมื่อเจอกับข้อมูลที่ท้าทาย บ่งชี้ว่าโครงสร้างที่มี conv layer น้อยมีความไม่เสถียรในช่วงต้นของการเทรนมากกว่า configuration ที่ซับซ้อนกว่า

**7.3 ความผันผวนของ Validation Accuracy ช่วงท้ายการเทรน**
`cfg4` แสดงการร่วงของ validation accuracy อย่างรุนแรงในช่วง epoch 12-13 (จาก ~0.79 ลงไปเกือบ 0.50) ก่อนจะฟื้นตัวกลับมา ซึ่งเป็นสัญญาณของ **overfitting ชั่วคราว** — โมเดลที่มีพารามิเตอร์มาก (conv 4 ชั้น + dense 256 neuron) มีแนวโน้มจดจำรายละเอียดเฉพาะของ training set มากเกินไปในบางช่วง กลไก EarlyStopping ที่ใช้ในการทดลองนี้ช่วยให้ระบบดึง weight จาก epoch ที่ดีที่สุดกลับมาใช้ แทนที่จะใช้ weight จาก epoch สุดท้ายที่อาจ overfit ไปแล้ว

**7.4 สรุปแนวโน้ม**
Configuration ที่มีความซับซ้อนระดับปานกลาง (conv layer 3 ชั้น, dense 128 neuron) ร่วมกับจำนวน epoch ที่เพียงพอ (20 epoch พร้อม EarlyStopping) ให้ผลลัพธ์ที่ดีที่สุดและมีเสถียรภาพสูงกว่าทั้ง configuration ที่เรียบง่ายเกินไป (`cfg1`, `cfg2`) และซับซ้อนเกินไป (`cfg4`)

## 8. สรุปผล (Conclusion)

การทดลองนี้แสดงให้เห็นว่า CNN สามารถจำแนกภาพใบหน้าจริงและภาพที่สร้างจาก GAN ได้ด้วยความแม่นยำ 86.83% บนชุดข้อมูลทดสอบ แม้จะใช้ข้อมูลเพียง 12,000 รูป (8.5% ของชุดข้อมูลต้นฉบับ) ผลการเปรียบเทียบ configuration ยืนยันหลักการสำคัญทางทฤษฎี Deep Learning ว่าทั้ง**ความลึกของโมเดล**และ**จำนวน epoch** ต้องได้รับการปรับให้เหมาะสมร่วมกัน (balance) ไม่ใช่ยิ่งมากยิ่งดีเสมอไป และเทคนิคอย่าง EarlyStopping มีบทบาทสำคัญในการควบคุม overfitting และรักษาประสิทธิภาพของโมเดลให้อยู่ในจุดที่ดีที่สุด

---

## ภาคผนวก: วิธีรันโค้ด

### ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

### เตรียม dataset
```bash
python prepare_dataset.py --source <path_ที่แตก_zip_จาก_Kaggle>/real_vs_fake/real-vs-fake --n_per_class 6000
```

### รัน pipeline หลัก (เทรน + เปรียบเทียบ 4 config)
```bash
cd classification
python main.py
```

### ทดสอบโมเดลกับภาพสุ่ม 4 รูป
```bash
python test_cnn.py
```

### ปรับแต่ง configuration ที่จะเปรียบเทียบ
แก้ไขที่ตัวแปร `CONFIGS` ในไฟล์ `classification/main.py`