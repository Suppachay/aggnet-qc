---
name: aggregate-qc-ai
description: |
  ใช้ skill นี้สำหรับทุก task ที่เกี่ยวข้องกับโปรเจค AI สำหรับ QC มวลรวมหยาบ (coarse aggregate / หินก่อสร้าง) โดยใช้ภาพถ่ายหินในการ predict %Passing จาก ASTM Sieve Analysis ตั้งแต่ data preparation, model training (PyTorch), inference, ไปจนถึง evaluation และ grading curve report.

  Trigger เสมอเมื่อ user พูดถึง: หินก่อสร้าง, มวลรวมหยาบ, sieve analysis, %Passing, ตะแกรง, grading curve, aggregate QC, AggNet, sample image + weight prediction, หรือ task ใดๆ ที่เกี่ยวกับโปรเจคนี้
---

# Aggregate QC AI — SKILL.md

## บริบทโปรเจค

เป้าหมายคือให้ AI สามารถ predict **%Passing** ผ่านตะแกรง 7 ขนาด (1", 3/4", 1/2", 3/8", #4, #8, Pan) จาก:

- **Input**: ภาพถ่ายตัวอย่างหิน 1 ใบ (single image) + น้ำหนักตัวอย่าง (weight_g, หน่วย g)
- **Output**: %Passing ทั้ง 7 ค่า (regression, ค่า 0–100 แต่ละช่อง)

**หมายเหตุ Dataset3:** ตัด `1inch` ออกเพราะ = 100% ทุก sample → model จึง predict จริง 6 sieve (3/4", 1/2", 3/8", #4, #8, Pan)

**Aggregate Type ที่รองรับ (2026-06-25):**
- `Aggregate 3_4inch` → Model A (6 sieves) — CBAM + Multi-Task
- `Aggregate 3_8inch` → Model B (3 sieves: No4/No8/Pan, hardcode 3/4"=0, 1/2"=100, 3/8"≈99.7) — CBAM only
- `Aggregate 1 inch`  → Model C (รอข้อมูล ≥ 4 samples จึงจะ train ได้)

---

## โค้ดปัจจุบัน: `aggnet_dataset3.py`

**Path:** `/root/AggNet/code/aggnet_dataset3.py`
**Dataset:** `/root/AggNet/data/dataset3/`
**Splits:** `/root/AggNet/data/dataset3/splits.json`
**Output:** `/root/AggNet/outputs/dataset3/`
**Models:** `/root/AggNet/models/dataset3/`

### Configuration หลัก (ณ 2026-06-25)

| Parameter | ค่า | หมายเหตุ |
|-----------|-----|----------|
| `BATCH_SIZE` | 8 | |
| `MAX_EPOCHS` | 500 | early stopping ช่วย |
| `PATIENCE` | 60 | |
| `WEIGHT_DECAY` | 1e-3 | |
| `WEIGHT_MIN/MAX` | 400–900g | range normalize weight |
| `FREEZE_EPOCHS` | 30 | Phase 1: train head only |
| `LR_HEAD` | 3e-4 | Phase 1 LR |
| `LR_FINETUNE` | 2e-5 | Phase 2 LR |

### Data Split (splits.json)

| Set | สัดส่วน | หมายเหตุ |
|-----|---------|----------|
| Train | 70% | source-aware stratified split |
| Val | 20% | ใช้ตอน training เพื่อ early stopping |
| Test | 10% | **held out ไม่ใช้ตอน train เลย** |

### Architecture: EfficientNet-B0 + CBAM + Multi-Task (ตั้งแต่ Run 08)

```
Image (224×224×3)
    ↓
EfficientNet-B0 features (pretrained ImageNet)
    ↓
CBAM (Channel Attention + Spatial Attention)   ← NEW: โฟกัสบริเวณสำคัญในภาพ
    ↓ AdaptiveAvgPool
1280-dim feature
    ↓ concat
[weight_norm(1) + agg_type_code(1)] = scalar(2-dim)
    ↓ 1282-dim fusion
    ├── Head A: %Passing (primary)
    │   Linear(1282→256) → ReLU → Dropout(0.3)
    │   Linear(256→64)   → ReLU
    │   Linear(64→n_sieves) → Sigmoid
    │
    ├── Head B: Gradation Class (auxiliary)         ← NEW: coarse/medium/fine
    │   Linear(1282→64) → ReLU → Dropout(0.2)
    │   Linear(64→3)    → CrossEntropy
    │
    └── Head C: Production Ratio (auxiliary)        ← NEW: % retain on 1/2" and 3/8"
        Linear(1282→64) → ReLU → Dropout(0.2)
        Linear(64→2)    → Sigmoid
```

**CBAM (Convolutional Block Attention Module):**
- **Channel Attention**: Squeeze-Excitation style — เรียนรู้ว่า feature channel ไหนสำคัญ (เช่น texture vs color)
- **Spatial Attention**: เรียนรู้ว่าตำแหน่งไหนในภาพสำคัญ (เช่น บริเวณเม็ดเล็ก vs เม็ดใหญ่)
- เพิ่ม parameter เพียง ~206K (5% ของ total)

**Multi-Task Learning (Model A เท่านั้น):**
- **Gradation Class**: coarse (1/2"<30%) / medium (30-55%) / fine (>55%) — บังคับ backbone เรียนรู้ขอบเขตที่ชัดเจน
- **Production Ratio**: % retain on 1/2" and 3/8" — บังคับเรียนรู้ความสัมพันธ์ระหว่าง sieve
- Loss: `L_passing + 0.3 × L_gradation + 0.3 × L_production`

**Two-Phase Training:**
- **Phase 1** (`FREEZE_EPOCHS=30`): Freeze backbone → train CBAM + heads only (LR=3e-4)
- **Phase 2** (ต่อจาก Phase 1): Partial unfreeze (last 2 EfficientNet blocks + CBAM + heads) → fine-tune (LR=2e-5)

**Aggregate type encoding:** `{'Aggregate 3_4inch': 0.0, 'Aggregate 3_8inch': 1.0, 'Aggregate 1 inch': 2.0}`

### Architecture ก่อนหน้า (เลิกใช้แล้ว)

1. **AggNet Custom CNN** (Run 01–04): scratch CNN, 128-dim, ไม่มี pretrained → predict ค่าเฉลี่ย
2. **EfficientNet-B0 + GAP** (Run 05–07): pretrained backbone, ไม่มี attention → underfitting บน 1/2", 3/8"

### SIEVE_COLS แยกตาม model

| Model | Filter | SIEVE_COLS | Model File |
|-------|--------|-----------|------------|
| A | `Aggregate 3_4inch` | `['3_4inch','1_2inch','3_8inch','No4','No8','Pan']` | `aggnet_34_best.pth` |
| B | `Aggregate 3_8inch` | `['No4','No8','Pan']` | `aggnet_38_best.pth` |
| C | `Aggregate 1 inch`  | `['3_4inch','1_2inch','3_8inch','No4','No8','Pan']` | `aggnet_1inch_best.pth` |

### Duplicate Detection (auto)

`clean_duplicates()` เรียกอัตโนมัติใน `load_records()` ทุกครั้ง:
- ตรวจ label ซ้ำทุก sieve column ภายใน agg_type เดียวกัน
- keep first occurrence, drop rest พร้อม warning
- **Dataset3 (2026-06-25):** ตัดออก 2 samples (093=051, 132=049) → เหลือ **145 samples**

---

## Dataset (2026-06-25)

| ประเภท | จำนวน (หลัง clean) | Split (train/val/test) |
|--------|-------------------|----------------------|
| Aggregate 3_4inch | **104 samples** | 73 / 21 / 10 |
| Aggregate 3_8inch | **40 samples** | 28 / 8 / 4 |
| Aggregate 1 inch  | **1 sample** | รอข้อมูลเพิ่ม |
| **รวม** | **145 samples** | |

### Production Ratio Analysis (Model A — % Individual Retain)

| Sieve | ASTM Range | Control Range | Data Mean | % In ASTM |
|-------|-----------|--------------|-----------|-----------|
| Retain 3/4" | 25–45% | 35–43% | 12.0% (std 9.1) | 6.7% |
| Retain 1/2" | 20–35% | 29–35% | 46.7% (std 13.3) | 18.3% |
| Retain 3/8" | 20–55% | 25–33% | 20.2% (std 6.2) | 55.8% |

**0% ของ samples ผ่าน ASTM ครบทั้ง 3 sieve** — ข้อมูลจริงในตลาดเกือบไม่มีที่ production ratio สมดุลตามมาตรฐาน

### %Passing Statistics (Model A)

| Sieve | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| 3/4" | 87.8 | 9.3 | 53.3 | 99.4 |
| 1/2" | 41.1 | 17.9 | 5.6 | 90.0 |
| 3/8" | 20.9 | 14.5 | 1.2 | 53.8 |
| No4 | 2.6 | 4.7 | 0.04 | 24.8 |
| No8 | 1.3 | 2.5 | 0.02 | 15.0 |
| Pan | 0.0 | 0.0 | 0.0 | 0.0 |

### Loss Function: `MultiTaskLoss` (Model A) / `MonotonicMSELoss` (Model B)

```python
# MultiTaskLoss (Model A)
loss = MonotonicMSE(passing_pred, target)
     + 0.3 × CrossEntropy(grad_logits, grad_class)
     + 0.3 × MSE(prod_pred, prod_target)

# MonotonicMSELoss
loss = MSE(pred, target) + 0.5 × mean(clamp(pred[:,i+1] - pred[:,i], min=0)²)
```

### Data Augmentation (train)

RandomHFlip, RandomVFlip, RandomRotation(20°), RandomResizedCrop(scale=0.75–1.0),
ColorJitter(b=0.5,c=0.5,s=0.4,h=0.15), RandomGrayscale(p=0.15),
GaussianBlur, RandomErasing(p=0.3)

### Dataset Format (labels.csv)

```
sample_id, 1inch, 3_4inch, 1_2inch, 3_8inch, No4, No8, Pan, weight_g, Aggregate Type, Source, Tested Date
1, 100.0, 76.03, 22.45, 8.37, 0.36, 0.32, 0.0, 500, Aggregate 3_4inch, ART CONCRETE, 30/9/2025
```

Image naming: `sample_id=1` → `Sample_001.jpg`

### Metrics ที่ใช้

| Metric | ความหมาย |
|--------|----------|
| MAE (%) | error เฉลี่ยทุก sieve-sample |
| Per-sieve Accuracy (±5%) | % cell ที่ error ≤ 5% |
| Sample Accuracy (±5%) | % sample ที่ผ่านทุกตะแกรง (QC pass/fail) |
| R² (3/4"–3/8") | เฉพาะ 3 sieve ใหญ่ที่มี variance สูงพอ |

### Output Files

| ไฟล์ | หน้าที่ |
|------|---------|
| `models/dataset3/aggnet_34_best.pth` | best weights — Model A (3_4inch) |
| `models/dataset3/aggnet_38_best.pth` | best weights — Model B (3_8inch) |
| `outputs/dataset3/history_3_4.png` | train/val loss curve — Model A |
| `outputs/dataset3/history_3_8.png` | train/val loss curve — Model B |
| `outputs/dataset3/metrics_summary_val.png` | MAE / R² / Accuracy per sieve (val) |
| `outputs/dataset3/metrics_summary_test.png` | MAE / R² / Accuracy per sieve (test) |
| `outputs/dataset3/gradation_val.png` | predicted vs actual curves (val) |
| `outputs/dataset3/gradation_test.png` | predicted vs actual curves (test) |

---

## Dataset Structure

```
/root/AggNet/
├── code/
│   ├── aggnet_dataset3.py       ← script หลัก (CBAM + Multi-Task, splits.json)
│   ├── aggnet_single_image.py   ← script เก่า (single-image)
│   ├── aggnet_multiview.py      ← script เก่า (multi-view)
│   └── app_aggnet_qc.py         ← Web App Flask (port 5000) — ใช้ CBAM+MT model
├── data/
│   ├── dataset3/                ← 147 samples + labels.csv + splits.json (ปัจจุบัน)
│   ├── source_split/            ← images จัดเป็น Model_A/B/C × train/val/test
│   ├── dataset_single/          ← 143 samples (Dataset เดิม)
│   └── dataset_multiview/       ← 21 samples × 3 views
├── models/
│   ├── dataset3/                ← aggnet_34_best.pth, aggnet_38_best.pth
│   ├── single/                  ← aggnet_single_best.pth
│   └── multiview/               ← aggnet_multiview_best.pth
├── outputs/
│   ├── dataset3/                ← PNG plots (val + test)
│   ├── single/                  ← PNG plots
│   └── multiview/               ← PNG plots
├── test_set/                    ← test_stone.jpg
├── skill.md
└── experiment_log.md
```

---

## Quick Start

```python
# Train (reads splits.json automatically)
python /root/AggNet/code/aggnet_dataset3.py

# Web App (CBAM + Multi-Task model)
python /root/AggNet/code/app_aggnet_qc.py   # http://0.0.0.0:5000

# Inference (uncomment ใน __main__)
result = predict(
    image_path  = "/root/AggNet/data/dataset3/Sample_006.jpg",
    weight_g    = 620,
    source_name = "ART CONCRETE COMPANY LIMITED"
)
```

---

## สิ่งที่ยังไม่ได้ทำ / TODO

### Priority สูง (ทำก่อน)
- [ ] **เพิ่มข้อมูล 3_4inch** โดยเฉพาะ "fine content สูง" (1/2">55%) และ "coarse มาก" (3/4"<70%) — ปัจจุบัน 1/2" MAE ~13%, 3/8" MAE ~11% ยังสูง
- [ ] **เพิ่มข้อมูล Aggregate 1 inch** ให้ถึง ≥ 4 samples เพื่อเปิด train Model C
- [x] ~~อัปเดต Web App~~ — ใช้ CBAM+MT model, Report 3/8" ตัด 1"/3/4" แล้ว

### Priority กลาง
- [ ] **k-fold cross-validation** แทน single val split เพื่อ metric ที่เชื่อถือได้มากขึ้น
- [ ] เพิ่ม `scaler.json` บันทึก weight normalization params (ปัจจุบัน hardcode WEIGHT_MIN/MAX)
- [ ] ASTM C33 gradation limits shading ใน grading curve plot
- [ ] Export batch inference เป็น CSV
- [ ] Tune multi-task loss weights (w_grad, w_prod) — ปัจจุบัน 0.3/0.3

### Priority ต่ำ
- [ ] เพิ่ม Isotonic Regression post-processing เพื่อบังคับ monotonic ตอน inference
- [ ] เพิ่ม Auto-detect agg_type จากภาพ (ตอนนี้ user ต้องเลือกเอง)
- [ ] Attention map visualization เพื่อดูว่า CBAM โฟกัสตรงไหน

### สำเร็จแล้ว ✅
- [x] Source-aware split (70/20/10 train/val/test, splits.json)
- [x] CBAM Attention Module
- [x] Multi-Task Learning (gradation class + production ratio)
- [x] Test set evaluation (held out)
- [x] Production Ratio Analysis

---

## หมายเหตุสำหรับ Claude

เมื่อ user ขอทำ task ใดในโปรเจคนี้:
1. **อ่าน experiment_log.md** ก่อนเสมอ เพื่อเข้าใจว่าทดลองอะไรไปแล้วบ้าง
2. **เสนอ code ที่ run ได้ทันที** — ไม่ใช่ pseudocode เท่านั้น
3. **แจ้ง assumption** เช่น สมมติว่า labels.csv มี column ชื่อ `weight_g`
4. **Dataset ปัจจุบัน 145 samples** (3_4inch: 104, 3_8inch: 40, 1inch: 1) — ทุก suggestion ต้องคำนึงถึง overfitting
5. **ตรวจสอบ monotonic constraint** ทุกครั้งที่มี prediction output
6. **อ้างอิง config ใน aggnet_dataset3.py** ก่อนแนะนำ hyperparameter
7. **Model ปัจจุบันใช้ CBAM + Multi-Task** — อย่าแนะนำ architecture ที่ถอยหลังกลับไป GAP ธรรมดา
8. **splits.json** กำหนด train/val/test — อย่าใช้ random split ใหม่เอง
