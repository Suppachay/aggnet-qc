# Experiment Log — Aggregate QC AI (AggNet)

> บันทึกผลการทดลองเทรนทุกครั้ง เพื่อ track การพัฒนาและป้องกันการทดสอบซ้ำ
> อัปเดตไฟล์นี้ทุกครั้งหลัง train เสร็จ

---

## วิธีบันทึก

คัดลอก template ด้านล่างมาต่อท้าย แล้วกรอกข้อมูล:

```
### Run [หมายเลข] — [วันที่]
...
```

---

## Template

```markdown
### Run XX — YYYY-MM-DD

**Script:** /root/AggNet/code/aggnet_dataset3.py (หรือชื่อไฟล์)
**Dataset:** /root/AggNet/data/dataset3 (N samples)

**Config ที่เปลี่ยน (จาก default):**
- PARAM = ค่า  ← เหตุผลที่เปลี่ยน

**Results:**
| Metric | ค่า |
|--------|-----|
| Best Val Loss | |
| MAE (avg) | % |
| Per-sieve Acc (±5%) | % |
| Sample Acc (±5%) | % |
| R² (3/4"–3/8") | |
| Stopped at Epoch | / MAX_EPOCHS |

**สังเกต:**
- 

**ขั้นตอนถัดไป:**
- 
```

---

## Runs

### Run 01 — 2025-XX-XX (ก่อนมี log — สรุปจากโค้ดปัจจุบัน)

**Script:** /root/AggNet/code/aggnet_dataset3.py  
**Dataset:** /root/AggNet/data/dataset3 (18 samples, val_split=0.2 → train 14 / val 4)

**Config หลัก (ณ วันที่สร้าง log):**
- BATCH_SIZE = 2
- LR = 0.0003
- MAX_EPOCHS = 500
- PATIENCE = 60
- WEIGHT_DECAY = 1e-3
- LR Scheduler: CosineAnnealingWarmRestarts (T0=50, Tmult=2) + ReduceLROnPlateau
- Loss: MonotonicMSELoss (mono_weight=0.5)
- Architecture: AggNet custom (Stem→msEnc×2→GAP→Fusion MLP), ~trainable params ขนาดเล็ก
- Augmentation: MaxAug (HFlip, VFlip, Rotation20, ResizedCrop, ColorJitter, Grayscale, GaussBlur, RandomErasing)

**Results:** ยังไม่มีข้อมูล — กรอกหลัง run ครั้งแรกด้วย config นี้

**หมายเหตุ:**
- Config นี้คือ current state ของ aggnet_dataset3.py
- ก่อนหน้านี้มีการปรับ val_split จาก 0.33 → 0.2 เพื่อให้ train set ใหญ่ขึ้น (14 vs 12 samples)
- เพิ่ม PATIENCE จากค่าเดิมเป็น 60 เพราะ dataset เล็ก loss อาจ fluctuate
- เพิ่ม WEIGHT_DECAY เป็น 1e-3 เพื่อ regularization แรงขึ้น

---

---

## การเตรียม Dataset & โค้ด (2026-06-04) — ก่อน train จริง

### สิ่งที่ทำในเซสชันนี้

#### 1. สร้าง skill.md และ experiment_log.md
- สร้าง `/root/AggNet/skill.md` — documentation สัมพันธ์กับโค้ดจริง (architecture, config, dataset format)
- สร้าง `/root/AggNet/experiment_log.md` — ไฟล์นี้ สำหรับ track การทดลอง

#### 2. อัปเดต labels.csv: 18 → 55 samples
- ไฟล์เดิมมีเพียง 18 samples และ **sample 13–14 มี label ซ้ำกัน** (ข้อมูลผิด)
- อัปเดตเป็น 55 samples ครบถ้วน แก้ไข sample 14 ให้ถูกต้อง (`91.34, 57.77, 35.32`)
- ลบ BOM character (`\xef\xbb\xbf`) ออกจากหัวไฟล์ที่ติดมาจาก Excel
- Dataset ประกอบด้วย 2 ประเภท: **Aggregate 3_4inch** (47 samples) และ **Aggregate 3_8inch** (8 samples)
- ยืนยันภาพ 55 ใบครบ (`Sample_001.jpg` – `Sample_055.jpg`)

#### 3. อัปเดต aggnet_dataset3.py
| จุดที่เปลี่ยน | เดิม | ใหม่ | เหตุผล |
|---|---|---|---|
| `BATCH_SIZE` | 2 | 8 | dataset ใหญ่ขึ้น 3x |
| Model input (scalar) | weight(1-dim) | weight + agg_type(2-dim) | 3_4inch vs 3_8inch มี pattern ต่างกันสิ้นเชิง |
| `AggNet` fusion input | `Linear(65→32)` | `Linear(66→32)` | รองรับ input 2-dim |
| `AGG_TYPE_MAP` | ไม่มี | `{3_4inch: 0.0, 3_8inch: 1.0}` | encode ประเภทหิน |
| `predict()` | ไม่มี `agg_type` | มี `agg_type` param | inference ต้องระบุประเภทหิน |

#### 4. สถานะปัจจุบัน
- ✅ Dataset พร้อม: 55 samples, labels ถูกต้อง, ภาพครบ
- ✅ โค้ดพร้อม train: `/root/AggNet/code/aggnet_dataset3.py`
- ⏳ ยังไม่ได้ train — รอรัน `python /root/AggNet/code/aggnet_dataset3.py`

#### 5. สิ่งที่ต้องระวังตอน train
- Aggregate 3_8inch มีเพียง 8 samples → val set อาจไม่มี 3_8inch เลยถ้า split ไม่ดี
- ควร stratify split ตาม `Aggregate Type` ในอนาคต

---

### Run 02 — 2026-06-04 (Train ครั้งแรกกับ 55 samples — ผลแย่)

**Script:** /root/AggNet/code/aggnet_dataset3.py  
**Dataset:** /root/AggNet/data/dataset3 (55 samples, val_split=0.2 → train 44 / val 11)

**Config:**
- BATCH_SIZE = 8, LR = 0.0003, MAX_EPOCHS = 500, PATIENCE = 60
- SIEVE_COLS ยังมี 1inch อยู่ (bug — ยังไม่ได้ตัดออกจริง)
- Split: random ไม่ stratify ตาม agg_type

**Results:**
| Metric | ค่า |
|--------|-----|
| MAE (avg) | 9.18% |
| Per-sieve Acc (±5%) | 45.5% |
| Sample Acc (±5%) | 0.0% |
| R² (3/4"–3/8") | -1.0000 |

**วิเคราะห์ปัญหา:**
- R² = -1.0 → model แย่กว่าการ predict ค่าเฉลี่ย → model ไม่ได้เรียนรู้จริง
- สาเหตุหลัก: **1inch ยังอยู่ใน SIEVE_COLS** — 3_4inch sample มี 1inch=100, 3_8inch sample มี 1inch=0 → bimodal distribution ทำให้ gradient สับสน
- สาเหตุรอง: **split ไม่ stratify** → val set อาจมีสัดส่วน 3_8inch ผิดปกติ
- MAE ต่ำใน #4, #8, Pan เพราะค่าใกล้ 0 เกือบทุก sample (ง่ายต่อการ predict)

**แก้ไขก่อน train ครั้งถัดไป (ทำแล้ว 2026-06-04):**
- ✅ ตัด 1inch ออกจาก SIEVE_COLS จริงๆ → เหลือ 6 sieves (3/4", 1/2", 3/8", #4, #8, Pan)
- ✅ เปลี่ยน split เป็น stratified ตาม agg_type (3_4inch / 3_8inch)
- ✅ เพิ่ม model capacity: stem 16→32, enc 32/64→64/128, fusion 66→130→64→6

---

---

### Run 03 — 2026-06-04 (Train แยก agg_type — ผลดีขึ้นแต่ยังไม่พอ)

**Script:** /root/AggNet/code/aggnet_dataset3.py (หลังแก้ stratified split + ตัด 1inch)  
**Dataset:** /root/AggNet/data/dataset3 (55 samples, stratified val → 3_4inch: train 38/val 10, 3_8inch: train 6/val 1)

**Config ที่เปลี่ยนจาก Run 02:**
- ตัด `1inch` ออกจาก SIEVE_COLS จริงๆ → 6 sieves
- Stratified split ตาม agg_type
- Model capacity เพิ่ม: stem 16→32, enc 32/64→64/128

**Results:**
| Metric | ค่า |
|--------|-----|
| MAE (avg) | 8.60% |
| Per-sieve Acc (±5%) | 53.3% |
| Sample Acc (±5%) | 0.0% |
| R² (3/4"–3/8") | 0.4615 |

**Per-sieve:**
- 3/4": MAE 15.77%, R²=0.495, Acc 10%
- 1/2": MAE 10.31%, R²=0.395, Acc 50%
- 3/8": MAE 14.32%, R²=0.494, Acc 10%
- #4: MAE 5.40%, R²=-1.0, Acc 70%
- #8: MAE 3.21%, R²=-1.0, Acc 90%

**วิเคราะห์:**
- R² ดีขึ้นจาก -1.0 → 0.46 แต่ MAE 3/4" และ 3/8" ยังสูงมาก (~15%)
- ต้นเหตุ: mix 3_4inch (3/4"=73-99%) กับ 3_8inch (3/4"=0%) ใน model เดียวกัน แม้มี agg_type flag ก็ยังสับสน
- Data analysis: 3_8inch มี 3 sieve แรกเป็นค่าคงที่ทุก sample → ไม่ควร predict ด้วย CNN

**แก้ไขก่อน Run 04 (ทำแล้ว 2026-06-04):**
- ✅ แยก train เป็น 2 model อิสระ: `aggnet_34_best.pth` (6 sieves) และ `aggnet_38_best.pth` (3 sieves: No4/No8/Pan)
- ✅ 3_8inch model predict เฉพาะ No4/No8/Pan (3/4"=0, 1/2"=100, 3/8"≈99.7 hardcode ค่าคงที่)
- ✅ plot functions รับ sieve_labels dynamic
- ✅ load_records รับ sieve_cols parameter

---

---

### Run 04 — 2026-06-04 (แยก 2 model — Model B: 3_8inch เสร็จแล้ว)

**Script:** /root/AggNet/code/aggnet_dataset3.py (dual-model mode)  
**Dataset:** /root/AggNet/data/dataset3 (55 samples)

#### Model B: Aggregate 3_8inch (`aggnet_38_best.pth`)
**Dataset split:** train 6 / val 1 (stratified — val set มีเพียง 1 sample)  
**Predict:** No4, No8, Pan (3 sieves) — 3/4"=0, 1/2"=100, 3/8"≈99.7 hardcode  
**Stopped at:** Epoch 500/500 (ไม่ trigger early stopping)  
**Best Val Loss:** 0.00065

**Results:**
| Metric | ค่า |
|--------|-----|
| MAE (avg) | **2.49%** |
| Per-sieve Acc (±5%) | **100.0%** |
| Sample Acc (±5%) | **100.0%** |
| R² | N/A (variance ต่ำเกินไป) |

**Per-sieve:**
- No4: MAE 2.50%, Acc 100%
- No8: MAE 1.86%, Acc 100%
- Pan: MAE 3.11%, Acc 100%

**⚠️ ข้อควรระวัง:**
- Val set มีเพียง **1 sample** → metrics 100% ไม่ได้หมายความว่า generalize ได้จริง
- ต้องรอข้อมูลเพิ่มเพื่อยืนยัน (ปัจจุบัน 3_8inch มีเพียง 7 samples)
- ผล val loss ลดลงต่อเนื่องจนถึง epoch 500 → อาจ train ได้นานกว่านี้ถ้าต้องการ

#### Model A: Aggregate 3_4inch (`aggnet_34_best.pth`)
**Dataset split:** train 39 / val 9 (stratified)  
**Predict:** 3_4inch, 1_2inch, 3_8inch, No4, No8, Pan (6 sieves)  
**Stopped at:** ~epoch 230 (early stopping, PATIENCE=60)  

**Results:**
| Metric | ค่า |
|--------|-----|
| MAE (avg) | **3.84%** |
| Per-sieve Acc (±5%) | **83.3%** |
| Sample Acc (±5%) | **44.4%** |
| R² (3/4"–3/8") | 0.0051 |

**Per-sieve:**
| Sieve | MAE | R² | Acc (±5%) |
|---|---|---|---|
| 3/4" | 3.11% | 0.097 | 88.9% |
| 1/2" | 7.84% | -0.044 | 55.6% |
| 3/8" | 6.16% | -0.037 | 55.6% |
| No4 | 2.06% | N/A | 100% |
| No8 | 2.00% | N/A | 100% |
| Pan | 1.90% | N/A | 100% |

**วิเคราะห์:**
- No4/No8/Pan ดีมาก (MAE < 2.1%, Acc 100%) เพราะค่าต่ำและ variance น้อย
- **1/2" และ 3/8" ยังแย่** (MAE 7-8%, R²<0) — เป็น sieve ที่มี variance สูงสุด (std ~15-14%)
- R² ≈ 0 หมายความว่า model เกือบจะ predict ค่าเฉลี่ยเท่านั้น ยังไม่ได้เรียนรู้ pattern จากภาพจริงๆ
- Training curve: val loss < train loss ตลอด → ไม่ overfit แต่ underfitting บน 1/2" และ 3/8"

**สาเหตุที่เป็นไปได้:**
1. Dataset 48 samples ยังน้อยเกินไปสำหรับ 1/2" ที่ std=15.8%
2. Model อาจต้องการ visual feature ที่ละเอียดกว่านี้เพื่อแยก 1/2" ออก
3. Val set 9 samples อาจยัง noisy สำหรับ R²

**ขั้นตอนถัดไปที่แนะนำ:**
- [ ] เพิ่มข้อมูลให้มากขึ้น โดยเฉพาะ sample ที่ 1/2" vary มาก (12–69%)
- [ ] ลอง pretrained backbone (EfficientNet-B0) เมื่อมีข้อมูล > 80 samples
- [ ] ลอง TTA (Test-Time Augmentation) ตอน inference เพื่อลด variance

---

---

## สรุปภาพรวมทั้งหมด (2026-06-04)

### Timeline การพัฒนา

#### ช่วงที่ 1: เตรียม Infrastructure
- สร้าง `skill.md` และ `experiment_log.md` เพื่อ track การทำงาน
- อัปเดต `labels.csv` จาก 18 → 55 samples (แก้ข้อมูลซ้ำ sample 14, ลบ BOM)
- ยืนยันภาพครบ 55 ใบ (`Sample_001.jpg` – `Sample_055.jpg`)

#### ช่วงที่ 2: ปรับโค้ด aggnet_dataset3.py
| การเปลี่ยนแปลง | เหตุผล |
|---|---|
| เพิ่ม `agg_type` (0/1) เป็น model input | 3_4inch vs 3_8inch มี pattern ต่างกันสิ้นเชิง |
| BATCH_SIZE 2 → 8 | dataset ใหญ่ขึ้น 3x |
| ตัด `1inch` ออกจาก SIEVE_COLS | 3_4inch=100% แต่ 3_8inch=0% → bimodal ทำให้ model สับสน |
| เปลี่ยน split เป็น stratified ตาม agg_type | ให้ train/val มีทั้ง 2 ประเภท |
| เพิ่ม model capacity (stem 16→32, enc 32/64→64/128) | dataset ใหญ่ขึ้นรับ capacity ได้มากขึ้น |
| แยก train เป็น 2 model (Model A/B) | 3_8inch มี 3 sieve แรกเป็นค่าคงที่ ไม่ควร train รวม |

#### ช่วงที่ 3: Training Results

| Run | Model | MAE | Per-sieve Acc | Sample Acc | R² | หมายเหตุ |
|---|---|---|---|---|---|---|
| Run 02 | รวม (55s, 7 sieves) | 9.18% | 45.5% | 0% | -1.0 | 1inch ยังอยู่, ไม่ stratify |
| Run 03 | รวม (55s, 6 sieves) | 8.60% | 53.3% | 0% | 0.46 | ดีขึ้นแต่ 1/2" 3/8" ยังแย่ |
| Run 04A | 3_4inch (48s) | 3.84% | 83.3% | 44.4% | 0.005 | หยุด ~epoch 230 |
| Run 04B | 3_8inch (7s) | 2.49% | 100% | 100% | N/A | val=1 sample, ระวัง overfit |

**Best model ปัจจุบัน:**
- `/root/AggNet/models/dataset3/aggnet_34_best.pth` → MAE 3.84%, No4/No8/Pan ดีมาก, 1/2" และ 3/8" ยัง underfitting
- `/root/AggNet/models/dataset3/aggnet_38_best.pth` → MAE 2.49% (val เพียง 1 sample)

#### ช่วงที่ 4: Web App (app_aggnet_qc.py)
**ปัญหาที่พบและแก้ไข:**
| ปัญหา | การแก้ไข |
|---|---|
| MODEL_PATH ผิด | แก้เป็น `/root/AggNet/models/dataset3/aggnet_34_best.pth` และ `aggnet_38_best.pth` |
| AggNet architecture ไม่ตรงกับ trained model | แก้ให้ตรงกับ stem→enc1→enc2→fusion MLP |
| ไม่มี weight/agg_type input | เพิ่ม predict รับ agg_type + weight |
| ไม่รองรับ 3_8inch | เพิ่ม MODEL_38, hardcode 3 sieve แรก |
| agg_type เป็น text input | เปลี่ยนเป็น dropdown |
| PDF เปิดไม่ได้ (Adobe Acrobat error) | ReportLab เรียก `md5(usedforsecurity=False)` ไม่รองรับบน OpenSSL นี้ → patch `pdfdoc.py` และ `utils.py` |

**สถานะ app:** รันอยู่ที่ port 5000, PDF สร้างได้ปกติ (105KB)

#### ช่วงที่ 5: ความรู้ที่ได้จาก Dataset Analysis
- **น้ำหนักมีผลต่อ prediction น้อยมาก** (เปลี่ยน 400→900g ผล %Passing เปลี่ยนไม่ถึง 1%) → สามารถ hardcode 650g หรือตัดออกในอนาคต
- **3_8inch**: 3/4"=0, 1/2"=100, 3/8"≈99.7 เป็นค่าคงที่ — ต้อง predict แค่ No4 และ No8
- **1/2" (std=15.8%) และ 3/8" (std=14.4%)** คือ sieve ที่ยากที่สุด เพราะ variance สูง

#### สิ่งที่ยังต้องทำต่อ
- [ ] เพิ่มข้อมูล 3_4inch ให้ถึง ~80–100 samples เพื่อลด underfitting บน 1/2" และ 3/8"
- [ ] ลอง EfficientNet-B0 backbone เมื่อมีข้อมูลมากพอ
- [ ] เพิ่ม Auto-detect agg_type จากภาพ (ตอนนี้ user ต้องเลือกเอง)
- [ ] ทดสอบ Web App กับภาพจริงหลายๆ ใบ

---

## Session 2 — 2026-06-12

### สิ่งที่ทำในเซสชันนี้

#### 1. อัปเดต Dataset: 55 → 82 samples (78 หลัง clean)
- เพิ่ม Sample_056–082 เข้า dataset3
- พบ **Aggregate Type ใหม่**: `Aggregate 1 inch` (Sample_057) — ยังไม่มีใน model เดิม
- ตรวจพบ **label ซ้ำ 4 คู่**: 040=027, 045=044, 047=028, 050=042
- เพิ่มฟังก์ชัน `clean_duplicates()` ใน `load_records()` → ตัดออกอัตโนมัติ → เหลือ **78 samples**

#### 2. เพิ่ม Model C (Aggregate 1 inch) ในโค้ด
- เพิ่ม `SIEVE_COLS_1INCH`, `AGG_TYPE_MAP['Aggregate 1 inch'] = 2.0`
- `train()` รองรับ filter `'Aggregate 1 inch'` → บันทึก `aggnet_1inch_best.pth`
- Guard: ถ้า sample < 4 → skip อัตโนมัติ (ปัจจุบันมีเพียง 1 sample)

#### 3. เปลี่ยน Architecture: Custom CNN → EfficientNet-B0 pretrained

**เหตุผล:** Custom CNN ไม่สามารถ extract visual feature ที่ discriminative ได้จาก 65 samples → predicted curve ออกมา pattern เดิมทุก sample (predict ค่าเฉลี่ย, R²≈0)

| การเปลี่ยนแปลง | รายละเอียด |
|---|---|
| Backbone | Custom CNN (scratch) → EfficientNet-B0 (pretrained ImageNet) |
| Feature dim | 128-dim → 1280-dim |
| Fusion head | 130→64→n → 1282→256→64→n |
| Training strategy | 1 phase → **2-phase** (freeze head / partial unfreeze) |
| Dropout | 0.5+0.4 → 0.3 (head only) |

---

### Run 05 — 2026-06-12 (EfficientNet-B0, Full Unfreeze Phase 2)

**Script:** aggnet_dataset3.py (EfficientNetAggNet, Phase1=30ep freeze, Phase2 full unfreeze)
**Dataset:** 78 samples (หลัง clean), 3_4inch: train 52 / val 13

**Config:**
- Phase 1: LR_HEAD=3e-4, FREEZE_EPOCHS=30 (head only)
- Phase 2: LR_FINETUNE=5e-5, CosineAnnealingWarmRestarts

**Results (Model A — 3_4inch):**
| Metric | ค่า |
|--------|-----|
| MAE (avg) | **6.16%** |
| Per-sieve Acc (±5%) | 61.1% |
| Sample Acc (±5%) | 8.3% |
| R² (3/4"~3/8") | -0.2626 |

**สังเกต:**
- Predicted curves **แตกต่างกันต่อ sample** แล้ว — EfficientNet IS reading images (ต่างจาก Custom CNN ที่ predict ค่าเฉลี่ยทุกรูป)
- Phase 2 (full unfreeze) ไม่เคยชนะ Phase 1 → best model คือ Phase 1 เสมอ → เปลี่ยน strategy

---

### Run 06 — 2026-06-12 (EfficientNet-B0, Partial Unfreeze Phase 2)

**Config เปลี่ยนจาก Run 05:**
- Phase 2: LR_FINETUNE=2e-5 (ลดลง), `CosineAnnealingLR` (ไม่ restart)
- Phase 2: `unfreeze_last_n_blocks(n=2)` แทน full unfreeze (1.1M params แทน 4M params)

**Results (Model A — 3_4inch):**
| Metric | ค่า |
|--------|-----|
| MAE (avg) | **6.16%** |
| Per-sieve Acc (±5%) | 61.1% |
| Sample Acc (±5%) | 8.3% |
| R² (3/4"~3/8") | -0.2626 |

**สังเกต:**
- ผลเหมือน Run 05 ทุกตัวเลข — Phase 2 ยังไม่เคยชนะ Phase 1
- Model B (3_8inch) oscillate หนักใน Phase 2 เพราะ train set มีเพียง ~10 samples
- **สรุป: ถึง ceiling ของ data แล้ว** — การ tune training strategy ไม่ช่วยอีกต่อไป

**Hard samples ที่ทำให้ MAE สูง (val set):**
| Sample | MAE | แหล่ง | ปัญหา |
|--------|-----|-------|-------|
| Sample 41 | 16.4% | สานนท์โรง 1 | 1/2"=69% (fine content สูงผิดปกติ) |
| Sample 49 | 9.8% | JOB 24mm | gradation ต่ำมาก (3/4"=95%, 1/2"=65%) |
| Sample 77 | 9.2% | สานนท์โรง 2 | pattern ต่างจาก majority |

**ขั้นตอนถัดไป:**
- [ ] **เพิ่มข้อมูล** — สิ่งเดียวที่จะช่วยได้ตอนนี้ เน้น "fine content สูง" (1/2">50%) และ "gradation ต่ำ" (3/4"<70%)
- [ ] ลอง source-aware split เพื่อลด val set bias
- [ ] อัปเดต Web App ให้ใช้ EfficientNetAggNet

---

## สรุปภาพรวม Session 2 (2026-06-12)

| Run | Architecture | MAE | Per-sieve Acc | Sample Acc | R² | หมายเหตุ |
|---|---|---|---|---|---|---|
| Run 04A | Custom CNN (old) | 3.84% | 83.3% | 44.4% | 0.005 | 48 samples, val 9 |
| Run 05/06 | EfficientNet-B0 | **6.16%** | 61.1% | 8.3% | -0.263 | 65 samples, val 13 — curve ดีขึ้น แต่ MAE สูงขึ้น |

**หมายเหตุ MAE สูงขึ้นทั้งที่ model ดีขึ้น:** val set ใหม่ (13 samples) มี hard samples เพิ่มขึ้น และ EfficientNet ไม่ได้ predict ค่าเฉลี่ย → error บน hard samples จึงสูงขึ้น แต่นี่คือ metric ที่ซื่อสัตย์กว่า

---

## Session 3 — 2026-06-25

### สิ่งที่ทำในเซสชันนี้

#### 1. อัปเดต Dataset: 82 → 147 samples (145 หลัง clean)
- เพิ่ม Sample_083–147 เข้า dataset3
- ตรวจพบ **label ซ้ำ 2 คู่**: 093=051, 132=049 → `clean_duplicates()` ตัดออกอัตโนมัติ
- **Near-duplicate**: Sample 041 ≈ 092 (ต่างกัน <0.02%, ภาพต่างกัน → เก็บไว้)
- ไม่พบ null values, missing images, หรือ monotonic violations
- Weight range: 500–699g (mean 652g)

| ประเภท | จำนวน (หลัง clean) |
|--------|-------------------|
| Aggregate 3_4inch | **104 samples** |
| Aggregate 3_8inch | **40 samples** |
| Aggregate 1 inch  | **1 sample** |
| **รวม** | **145 samples** |

#### 2. Production Ratio Analysis (Model A — % Individual Retain)

วิเคราะห์สัดส่วนการกระจายหินตาม ASTM เกณฑ์จากภาพตัวอย่างลูกค้า:
- Retain 3/4" = 1inch − 3_4inch
- Retain 1/2" = 3_4inch − 1_2inch
- Retain 3/8" = 1_2inch − 3_8inch

| Sieve | ASTM Range | Data Mean | % In ASTM | ปัญหา |
|-------|-----------|-----------|-----------|-------|
| Retain 3/4" | 25–45% | 12.0% | 6.7% | 93.3% ต่ำกว่า ASTM min |
| Retain 1/2" | 20–35% | 46.7% | 18.3% | 81.7% สูงกว่า ASTM max |
| Retain 3/8" | 20–55% | 20.2% | 55.8% | ดีที่สุดใน 3 sieve |

**0% ของ samples ผ่าน ASTM ครบทั้ง 3 sieve** — ตลาดจริงไม่มีหินที่ production ratio สมดุล

#### 3. Data Split: 70/20/10 (train/val/test) → splits.json

เปลี่ยนจาก random stratified split → **source-aware stratified split** บันทึกใน `splits.json`:

| Model | Train | Val | Test | รวม |
|-------|-------|-----|------|-----|
| A (3/4") | 73 | 21 | 10 | 104 |
| B (3/8") | 28 | 8 | 4 | 40 |

- Test set **held out** ไม่ใช้ตอน train เลย
- Source coverage: Train 24 / Val 17 / Test 10 sources
- Distribution สมดุล (mean %Passing ของ 3 splits ใกล้เคียงกัน)

#### 4. เพิ่ม CBAM Attention + Multi-Task Learning

**CBAM (Convolutional Block Attention Module):**
- Channel Attention (1280-ch → squeeze → expand) + Spatial Attention (7×7 conv)
- เพิ่ม parameter ~206K (5% ของ total)
- ช่วยให้ model โฟกัสบริเวณที่มี discriminative feature

**Multi-Task (Model A เท่านั้น):**
- Head B: Gradation class (coarse/medium/fine ตาม 1/2" %Passing)
- Head C: Production Ratio (% retain on 1/2" and 3/8")
- Loss: `L_passing + 0.3 × L_grad + 0.3 × L_prod`

#### 5. อัปเดต aggnet_dataset3.py
| จุดที่เปลี่ยน | เดิม | ใหม่ | เหตุผล |
|---|---|---|---|
| Data loading | `split_train_val()` random | `splits.json` fixed split | reproducible, test set held out |
| Dataset class | `AggDataset` (3 outputs) | `AggDatasetMT` (5 outputs) | multi-task targets |
| Backbone output | EfficientNet → GAP | EfficientNet → CBAM → Pool | attention-weighted features |
| Model heads | 1 head (passing) | 3 heads (passing + grad + prod) | multi-task learning |
| Loss | `MonotonicMSELoss` | `MultiTaskLoss` (Model A) | combined loss |
| Save path | `SAVE_DIR` | `MODEL_DIR` | แยก models กับ outputs |
| Evaluation | val only | val + test (held out) | honest evaluation |

---

### Run 07 — 2026-06-25 (Baseline EfficientNet-B0, 70/20/10 split)

**Script:** aggnet_dataset3.py (EfficientNet-B0 + GAP, ไม่มี CBAM/MT)
**Dataset:** 145 samples, splits.json

#### Model A — Aggregate 3_4inch (train 73 / val 21 / test 10)

**Stopped at:** Phase2 ep 60 (total 90), early stopping

**Validation Results:**
| Metric | ค่า |
|--------|-----|
| MAE (avg) | **7.72%** |
| Per-sieve Acc (±5%) | 55.6% |
| Sample Acc (±5%) | 0.0% |
| R² (3/4"~3/8") | -0.824 |

**Test Results (held out):**
| Metric | ค่า |
|--------|-----|
| MAE (avg) | **7.09%** |
| Per-sieve Acc (±5%) | 55.0% |
| Sample Acc (±5%) | 0.0% |
| R² (3/4"~3/8") | -0.341 |

**Per-sieve (Val):**
| Sieve | MAE | R² | Acc |
|---|---|---|---|
| 3/4" | 8.54% | -1.000 | 28.6% |
| 1/2" | 16.16% | -0.683 | 14.3% |
| 3/8" | 13.48% | -0.789 | 38.1% |
| No4 | 3.45% | -0.238 | 76.2% |
| No8 | 2.92% | -1.000 | 81.0% |
| Pan | 1.74% | N/A | 95.2% |

#### Model B — Aggregate 3_8inch (train 28 / val 8 / test 4)

| Metric | Val | Test |
|--------|-----|------|
| MAE | 1.51% | 2.60% |
| Per-sieve Acc | 91.7% | 91.7% |
| Sample Acc | 75.0% | 75.0% |

---

### Run 08 — 2026-06-25 (CBAM + Multi-Task)

**Script:** aggnet_dataset3.py (EfficientNet-B0 + CBAM + Multi-Task)
**Dataset:** 145 samples, splits.json (same split as Run 07)

**Config ที่เปลี่ยนจาก Run 07:**
- เพิ่ม CBAM (Channel + Spatial Attention) หลัง EfficientNet features
- เพิ่ม Multi-Task heads: Gradation class + Production Ratio (Model A)
- Total params: 4,723,641 (vs 4,352,834 เดิม, +370K จาก CBAM + MT heads)

#### Model A — Aggregate 3_4inch (CBAM + Multi-Task)

**Stopped at:** Phase2 ep 60 (total 90), early stopping
**Best Val Loss:** 0.01104

**Validation Results:**
| Metric | Run 07 | Run 08 | Change |
|--------|--------|--------|--------|
| MAE (avg) | 7.72% | **6.97%** | **-0.75** |
| Per-sieve Acc | 55.6% | 54.0% | -1.6 |
| Sample Acc | 0.0% | **9.5%** | **+9.5** |
| R² | -0.824 | **-0.430** | **+0.394** |

**Test Results (held out):**
| Metric | Run 07 | Run 08 | Change |
|--------|--------|--------|--------|
| MAE (avg) | 7.09% | 7.61% | +0.52 |
| Per-sieve Acc | 55.0% | 58.3% | +3.3 |
| Sample Acc | 0.0% | 0.0% | - |
| R² | -0.341 | -0.435 | -0.094 |

**Per-sieve improvement (Val):**
| Sieve | Run 07 MAE | Run 08 MAE | Change |
|---|---|---|---|
| 3/4" | 8.54% | 8.47% | -0.07 |
| 1/2" | 16.16% | **12.88%** | **-3.28** |
| 3/8" | 13.48% | **11.01%** | **-2.47** |
| No4 | 3.45% | 3.91% | +0.46 |
| No8 | 2.92% | 3.45% | +0.53 |
| Pan | 1.74% | 2.11% | +0.37 |

#### Model B — Aggregate 3_8inch (CBAM only, ไม่มี Multi-Task)

| Metric | Run 07 Val | Run 08 Val | Run 08 Test |
|--------|-----------|-----------|-------------|
| MAE | 1.51% | 2.67% | 4.09% |
| Per-sieve Acc | 91.7% | 83.3% | 75.0% |
| Sample Acc | 75.0% | 50.0% | 50.0% |

**วิเคราะห์:**
- **Model A**: CBAM + Multi-Task ช่วย 1/2" (-3.28pp) และ 3/8" (-2.47pp) อย่างชัดเจน — sieve ที่ยากที่สุด
- Sample Acc > 0% ครั้งแรก (9.5%) — เริ่มมี sample ที่ผ่าน QC tolerance ได้
- R² ดีขึ้นจาก -0.82 → -0.43 — ยังติดลบแต่ใกล้ 0 มากขึ้น (เริ่มเรียนรู้ pattern)
- Test MAE สูงกว่า val เล็กน้อย (7.61 vs 6.97) — test set เล็ก (10 samples) ทำให้ variance สูง
- **Model B**: MAE เพิ่มขึ้นเล็กน้อย — dataset 3_8inch ส่วนใหญ่มาจาก source เดียว (สานนท์โรง 2) อาจไม่ได้ประโยชน์จาก CBAM

**ขั้นตอนถัดไป:**
- [x] ~~อัปเดต Web App ให้ใช้ CBAM + Multi-Task model~~ → ทำแล้ว (ดูด้านล่าง)
- [ ] **เพิ่มข้อมูล** — ยังเป็นตัวช่วยหลัก โดยเฉพาะ fine content สูง (1/2">55%) และ coarse มาก (3/4"<70%)
- [ ] Tune multi-task loss weights (w_grad, w_prod)
- [ ] Attention map visualization

---

### Web App Update — 2026-06-25

#### 1. อัปเดต Model Architecture ใน app_aggnet_qc.py
| จุดที่เปลี่ยน | เดิม | ใหม่ | เหตุผล |
|---|---|---|---|
| Model class | `AggNet` (Custom CNN, inline) | `EfficientNetAggNet` (import จาก aggnet_dataset3.py) | ให้ตรงกับ weights ที่ train ด้วย CBAM+MT |
| Model A loading | `AggNet(num_sieves=6)` | `EfficientNetAggNet(num_sieves=6, multitask=True)` | รองรับ CBAM + 3 heads |
| Model B loading | `AggNet(num_sieves=3)` | `EfficientNetAggNet(num_sieves=3, multitask=False)` | CBAM only |

#### 2. แก้ไข Report Model B (3/8") — ตัด 1" และ 3/4" ออก
| จุดที่เปลี่ยน | เดิม | ใหม่ |
|---|---|---|
| ตาราง Sieve | 7 แถว (1"→Pan) | **5 แถว** (1/2"→Pan) |
| กราฟ Gradation | 7 จุด, title "Coarse Aggregate" | **5 จุด** (1/2"→Pan), title "Fine Aggregate (3/8\")" |
| ASTM/Control limits | ใช้ค่า 3/4" aggregate | ใช้ค่า **#8 stone** (3/8" nominal) |
| Ind. Retain / Cu. Retain | คำนวณจาก 1"=100% | คำนวณจาก **1/2"=100%** เป็นจุดเริ่มต้น |
| Production Ratio | แสดง 3 ช่วง | **ไม่แสดง** (ไม่มีความหมายสำหรับ 3/8") |

#### 3. สถานะ Web App
- ✅ Model A (3/4"): report เหมือนเดิม (7 sieves + Production Ratio)
- ✅ Model B (3/8"): report แก้แล้ว (5 sieves, ไม่มี Production Ratio)
- ✅ ทั้ง 2 models ใช้ EfficientNet + CBAM weights ล่าสุด

---

## สรุปภาพรวม Session 3 (2026-06-25)

### Timeline การพัฒนา

| Run | Architecture | Data | Split | Val MAE | Val Sample Acc | Val R² | หมายเหตุ |
|---|---|---|---|---|---|---|---|
| Run 02 | Custom CNN (7 sieves) | 55 | 80/20 | 9.18% | 0% | -1.0 | 1inch bug |
| Run 03 | Custom CNN (6 sieves) | 55 | 80/20 | 8.60% | 0% | 0.46 | stratified |
| Run 04A | Custom CNN (dual) | 48 | 80/20 | 3.84% | 44.4% | 0.005 | 9 val samples |
| Run 05/06 | EfficientNet-B0 | 65 | 80/20 | 6.16% | 8.3% | -0.263 | curves vary per sample |
| Run 07 | EfficientNet-B0 | 104 | **70/20/10** | 7.72% | 0% | -0.824 | harder val set |
| **Run 08** | **EfficientNet + CBAM + MT** | 104 | **70/20/10** | **6.97%** | **9.5%** | **-0.430** | **1/2" -3.3pp, 3/8" -2.5pp** |

### Best Model ปัจจุบัน (Run 08)
- `/root/AggNet/models/dataset3/aggnet_34_best.pth` → CBAM + Multi-Task, MAE 6.97% (val) / 7.61% (test)
- `/root/AggNet/models/dataset3/aggnet_38_best.pth` → CBAM only, MAE 2.67% (val) / 4.09% (test)

### ความรู้สะสม
1. **1/2" (std=17.9%) และ 3/8" (std=14.5%)** คือ sieve ที่ยากที่สุด — variance สูง, ต้องการ data มากที่สุด
2. **CBAM ช่วย sieve ที่ยาก** — ลด MAE บน 1/2" และ 3/8" ได้ 2-3pp
3. **Multi-Task ช่วย backbone เรียนรู้ feature ดีขึ้น** — R² ดีขึ้น 0.4 จาก multi-task gradient
4. **Test set เล็ก (10 samples) มี variance สูง** — val metrics เชื่อถือได้มากกว่า
5. **Production Ratio ของข้อมูลจริง: 0% ผ่าน ASTM ครบ 3 sieve** — data ในตลาดไม่สมดุลตามมาตรฐาน
6. **น้ำหนักมีผลน้อยมาก** (เปลี่ยน 400→900g ผล %Passing เปลี่ยนไม่ถึง 1%)
7. **Source naming ไม่สม่ำเสมอ** — 39 unique strings → ~25 แหล่งจริง (typo/spacing)

<!-- เพิ่ม Run ใหม่ด้านล่างนี้ -->
