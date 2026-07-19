"""
AggNet - Single Image Regression (6 Sieves ASTM C 136)
=======================================================
Input  : 1 ภาพรวมหินก่อนร่อน (3 channels RGB)
Output : % Passing ของ 6 ตะแกรง ASTM C 136

Dataset structure:
    DATA_DIR/
    ├── labels.csv
    ├── Sample_001.jpg   ← ภาพรวมหินก่อนร่อน
    ├── Sample_002.jpg
    ├── Sample_003.jpg
    └── ...

labels.csv format (Dataset3):
    sample_id, 1inch, 3_4inch, 1_2inch, 3_8inch, No4, No8, Pan, weight_g, Aggregate Type, Source, Tested Date
    1, 100.0, 88.6, 34.8, 11.1, 0.2, 0.2, 0.0, 620, Aggregate 3_4inch, ART CONCRETE, 2025-09-30

หมายเหตุ:
    - 1inch ตัดออกเพราะ = 100% ทุก sample (ไม่มีประโยชน์ในการเรียนรู้)
    - weight_g บันทึกไว้ใน CSV แต่ไม่ใช้เป็น input ของ model
      (อนาคตสามารถเพิ่มเป็น feature ได้)
    2, 85.2, 16.9, 6.0,  0.3, 0.3, 0.0, Aggregate 3_4inch, BNT CONCRETE, 2025-10-01
    ...

หมายเหตุ:
    - sample_id=1 → ภาพชื่อ Sample_001.jpg (รองรับ .jpg .jpeg .png .bmp .tiff)
    - ตัด 1inch ออก เพราะ % passing = 100% ทุก sample (ไม่มีประโยชน์ในการเรียนรู้)
    - ค่า % Passing ช่วง 0.0 – 100.0
"""

import os
import copy
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR    = "/workspace/AggNet/data/dataset3"
LABELS_CSV  = os.path.join(DATA_DIR, "labels.csv")
SPLITS_JSON = os.path.join(DATA_DIR, "splits.json")
SAVE_DIR    = "/workspace/AggNet/outputs/dataset3"
MODEL_DIR   = "/workspace/AggNet/models/dataset3"
IMG_SIZE   = (224, 224)   # H x W
BATCH_SIZE   = 8
LR           = 0.0003
MAX_EPOCHS   = 500
PATIENCE     = 60
SEED         = 42

# ── LR Scheduler config ──
LR_FACTOR    = 0.5
LR_PATIENCE  = 20         # รอนานขึ้นก่อนลด LR
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-3       # regularization แรงขึ้นเพื่อป้องกัน overfit

# ── ASTM C 136 Sieve config แยกตาม Aggregate Type ──
# 3_4inch  : 3/4" vary (73-99%), 1/2" vary (12-69%), 3/8" vary (3-54%), No4/No8/Pan vary
# 3_8inch  : 3/4"=0 (const), 1/2"=100 (const), 3/8"≈99.7 (const), No4/No8/Pan vary
# 1inch    : 1"≈100% (const), 3/4" vary, 1/2"/3/8"/No4/No8/Pan vary
SIEVE_COLS_34     = ['3_4inch', '1_2inch', '3_8inch', 'No4', 'No8', 'Pan']
SIEVE_COLS_38     = ['No4', 'No8', 'Pan']   # 3 sieve แรก = constant ไม่ต้อง predict
SIEVE_COLS_1INCH  = ['3_4inch', '1_2inch', '3_8inch', 'No4', 'No8', 'Pan']  # 1"≈100% ตัดออก

SIEVE_COLS   = SIEVE_COLS_34   # default (สำหรับ train loop ทั่วไป)
SIEVE_SIZES  = [19.00, 12.50, 9.50, 4.75, 2.36, 0.001]
SIEVE_LABELS = ['3/4"\n19.0', '1/2"\n12.5', '3/8"\n9.50',
                '#4\n4.75',  '#8\n2.36',   'Pan']
NUM_SIEVES   = len(SIEVE_COLS_34)   # 6 (max)

# ── EfficientNet-B0 Two-Phase Training ──
FREEZE_EPOCHS = 30    # Phase 1: freeze backbone, train head only
LR_HEAD       = 3e-4  # Phase 1 LR (head only)
LR_FINETUNE   = 2e-5  # Phase 2 LR (full fine-tune) — ต่ำลงเพื่อหยุด oscillation

# ── Weight normalization ──
# normalize weight_g เป็น 0-1 โดย assume range 400–900g
WEIGHT_MIN = 400.0
WEIGHT_MAX = 900.0

# ── Aggregate type encoding ──
# 0 = Aggregate 3_4inch, 1 = Aggregate 3_8inch, 2 = Aggregate 1 inch
AGG_TYPE_MAP = {
    'Aggregate 3_4inch': 0.0,
    'Aggregate 3_8inch': 1.0,
    'Aggregate 1 inch':  2.0,
}

# ── ชื่อภาพ format ──
# sample_id=1 → Sample_001.jpg
IMG_PREFIX = "Sample"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1.  DATASET
# ─────────────────────────────────────────────
class AggDataset(Dataset):
    """
    1 sample = 1 ภาพรวมหินก่อนร่อน + label vector (% passing 6 ค่า)
    Input tensor shape: (3, H, W)  RGB
    """
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec      = self.records[idx]
        img_path = rec['img_path']
        label    = rec['label']    # shape (7,) ค่า 0.0–1.0
        weight_g = rec['weight_g'] # น้ำหนักตัวอย่าง (g)
        agg_type = rec['agg_type'] # ประเภทหิน

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # normalize weight เป็น 0-1
        weight_norm = (weight_g - WEIGHT_MIN) / (WEIGHT_MAX - WEIGHT_MIN)
        weight_norm = float(np.clip(weight_norm, 0.0, 1.0))

        # encode agg_type เป็น 0/1
        agg_code = AGG_TYPE_MAP.get(agg_type, 0.0)

        return (img,
                torch.tensor([weight_norm, agg_code], dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))


def find_image(data_dir, sample_id):
    """หาไฟล์ภาพของ sample_id รองรับหลาย extension"""
    base = Path(data_dir) / f"{IMG_PREFIX}_{sample_id:03d}"
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        p = Path(str(base) + ext)
        if p.exists():
            return p
    return None


def clean_duplicates(df):
    """
    ตรวจสอบและลบ sample ที่มี label ซ้ำกันทุก sieve column ภายใน agg_type เดียวกัน
    - keep first occurrence, drop the rest
    - คืน (df_clean, dup_ids) โดย dup_ids คือ list ของ sample_id ที่ถูกตัดออก
    """
    sieve_check_cols = ['3_4inch', '1_2inch', '3_8inch', 'No4', 'No8', 'Pan']
    available_cols   = [c for c in sieve_check_cols if c in df.columns]

    dup_ids   = []
    keep_mask = pd.Series(True, index=df.index)

    for agg_type, grp in df.groupby('Aggregate Type'):
        dup_mask = grp.duplicated(subset=available_cols, keep='first')
        dup_rows = grp[dup_mask]
        if not dup_rows.empty:
            non_dup = grp[~dup_mask]
            for idx, row in dup_rows.iterrows():
                sid = int(row['sample_id'])
                dup_ids.append(sid)
                keep_mask.loc[idx] = False
                # หา original sample ที่มีค่าเดียวกันจริงๆ
                match = non_dup[
                    (non_dup[available_cols] == row[available_cols]).all(axis=1)
                ]
                ref_id = int(match.iloc[0]['sample_id']) if not match.empty else -1
                print(f"  [DUPLICATE] Sample_{sid:03d} มี label ซ้ำกับ Sample_{ref_id:03d}"
                      f" ({agg_type}) → ตัดออก")

    df_clean = df[keep_mask].reset_index(drop=True)
    if dup_ids:
        print(f"  [CLEAN] ตัด {len(dup_ids)} sample ออก: {[f'Sample_{i:03d}' for i in dup_ids]}")
        print(f"  [CLEAN] เหลือ {len(df_clean)} samples จาก {len(df)} samples")
    else:
        print("  [CLEAN] ไม่พบ label ซ้ำ — ใช้ข้อมูลครบทุก sample")

    return df_clean, dup_ids


def load_records(data_dir, labels_csv, sieve_cols=None, sample_ids=None):
    """
    Load records from labels.csv.
    sample_ids: if provided, only load these sample IDs (for split-based loading)
    """
    if sieve_cols is None:
        sieve_cols = SIEVE_COLS_34
    df = pd.read_csv(labels_csv)
    df.columns = df.columns.str.strip()
    df['Aggregate Type'] = df['Aggregate Type'].str.strip()

    print(f"  [LOAD] Raw samples in CSV: {len(df)}")

    # Remove duplicates
    df, _ = clean_duplicates(df)

    # Filter by sample_ids if provided
    if sample_ids is not None:
        df = df[df['sample_id'].isin(sample_ids)]
        print(f"  [FILTER] Selected {len(df)} samples from split")

    records = []
    missing = []

    for _, row in df.iterrows():
        sid      = int(row['sample_id'])
        img_path = find_image(data_dir, sid)

        if img_path is None:
            missing.append(f"Sample_{sid:03d}.*")
            continue

        pct      = np.array([float(row[c]) for c in sieve_cols], dtype=np.float32)
        label    = pct / 100.0
        weight_g = float(row['weight_g']) if 'weight_g' in row.index else 0.0

        records.append({
            'sample_id': sid,
            'img_path':  img_path,
            'label':     label,
            'weight_g':  weight_g,
            'source':    str(row.get('Source', '')),
            'agg_type':  str(row.get('Aggregate Type', '')),
        })

    if missing:
        print(f"[WARNING] Images not found: {missing}")

    print(f"  Loaded {len(records)} samples")
    return records


def load_splits(splits_json, model_key):
    """Load train/val/test sample IDs from splits.json"""
    with open(splits_json, 'r') as f:
        splits = json.load(f)
    s = splits[model_key]
    print(f"  [SPLIT] {model_key}: train={len(s['train'])} val={len(s['val'])} test={len(s['test'])}")
    return s['train'], s['val'], s['test']


# ─────────────────────────────────────────────
# 2.  TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms(train=True):
    if train:
        # Maximum augmentation — ชดเชย dataset เพียง 18 samples
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                   saturation=0.4, hue=0.15),
            transforms.RandomGrayscale(p=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────
# 3.  MODEL  (EfficientNet-B0 + CBAM + Multi-Task)
# ─────────────────────────────────────────────

# ── 3a. CBAM (Convolutional Block Attention Module) ──
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = x.mean(dim=[2, 3])
        mx  = x.amax(dim=[2, 3])
        att = torch.sigmoid(self.fc(avg) + self.fc(mx))
        return x * att.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention: which regions of the image matter"""
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """Channel + Spatial Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ── 3b. Gradation class labels (auto-generated from %Passing) ──
NUM_GRAD_CLASSES = 3   # coarse=0, medium=1, fine=2

def assign_gradation_class(label_vec, sieve_cols):
    """
    Classify sample based on 1/2" %Passing:
      coarse : 1/2" < 30%  (เม็ดหยาบ ส่วนใหญ่ค้างบน 1/2")
      medium : 30% ≤ 1/2" < 55%
      fine   : 1/2" ≥ 55%  (เม็ดละเอียด ผ่าน 1/2" เยอะ)
    """
    if '1_2inch' in sieve_cols:
        idx = sieve_cols.index('1_2inch')
        pct = label_vec[idx] * 100.0
    else:
        return 1  # default medium for Model B
    if pct < 30.0:
        return 0  # coarse
    elif pct < 55.0:
        return 1  # medium
    else:
        return 2  # fine


# ── 3c. Updated Dataset (returns gradation class + production ratio) ──
class AggDatasetMT(Dataset):
    """
    Multi-task dataset: returns (image, scalar, label, grad_class, prod_ratio)
    """
    def __init__(self, records, transform=None, sieve_cols=None):
        self.records    = records
        self.transform  = transform
        self.sieve_cols = sieve_cols or SIEVE_COLS_34

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec['img_path']).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = rec['label']
        weight_norm = float(np.clip(
            (rec['weight_g'] - WEIGHT_MIN) / (WEIGHT_MAX - WEIGHT_MIN), 0.0, 1.0))
        agg_code = AGG_TYPE_MAP.get(rec['agg_type'], 0.0)
        scalar = torch.tensor([weight_norm, agg_code], dtype=torch.float32)
        label_t = torch.tensor(label, dtype=torch.float32)

        # Gradation class
        grad_cls = assign_gradation_class(label, self.sieve_cols)

        # Production ratio (% individual retain, normalized 0-1)
        pct = label * 100.0
        if len(pct) >= 3 and '3_4inch' in self.sieve_cols:
            retain_12 = (pct[0] - pct[1]) / 100.0   # retain on 1/2"
            retain_38 = (pct[1] - pct[2]) / 100.0   # retain on 3/8"
            prod_ratio = torch.tensor([
                max(0.0, retain_12),
                max(0.0, retain_38),
            ], dtype=torch.float32)
        else:
            prod_ratio = torch.zeros(2, dtype=torch.float32)

        return (img, scalar, label_t,
                torch.tensor(grad_cls, dtype=torch.long),
                prod_ratio)


# ── 3d. Model with CBAM + Multi-Task Heads ──
class EfficientNetAggNet(nn.Module):
    """
    EfficientNet-B0 + CBAM Attention + Multi-Task Heads
    ════════════════════════════════════════════════════
    Architecture:
        Image  → EfficientNet-B0 features → CBAM → Weighted Pooling → 1280-dim
        Scalar → [weight_norm, agg_type_code] → 2-dim
        Fusion → [1280+2] = 1282-dim
            ├─ Head A: %Passing regression (n_sieves) → Sigmoid
            ├─ Head B: Gradation class (3 classes)
            └─ Head C: Production Ratio (2 values) → Sigmoid
    """
    def __init__(self, num_sieves=NUM_SIEVES, multitask=True):
        super().__init__()
        self.multitask = multitask

        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features        # → (batch, 1280, 7, 7)
        self.cbam     = CBAM(1280, reduction=16)
        self.pool     = nn.AdaptiveAvgPool2d(1)   # → (batch, 1280, 1, 1)

        feat_dim = 1280 + 2

        # Head A: %Passing regression (primary task)
        self.head_passing = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_sieves),
            nn.Sigmoid(),
        )

        if multitask:
            # Head B: Gradation class (coarse/medium/fine)
            self.head_gradation = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(64, NUM_GRAD_CLASSES),
            )
            # Head C: Production Ratio (retain on 1/2" and 3/8")
            self.head_production = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(64, 2),
                nn.Sigmoid(),
            )

    @property
    def fusion(self):
        return self.head_passing

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n=2):
        self.freeze_backbone()
        total_blocks = len(self.features)
        for i, block in enumerate(self.features):
            if i >= total_blocks - n:
                for p in block.parameters():
                    p.requires_grad = True

    def _extract(self, img, scalar):
        x = self.features(img)          # (batch, 1280, 7, 7)
        x = self.cbam(x)                # attention-weighted
        x = self.pool(x)                # (batch, 1280, 1, 1)
        x = x.view(x.size(0), -1)      # (batch, 1280)
        return torch.cat([x, scalar], dim=1)  # (batch, 1282)

    def forward(self, img, scalar):
        feat = self._extract(img, scalar)
        passing = self.head_passing(feat)

        if self.multitask and self.training:
            grad_logits = self.head_gradation(feat)
            prod_ratio  = self.head_production(feat)
            return passing, grad_logits, prod_ratio

        return passing


# ── Legacy alias for inference/web app compatibility ──
AggNet = EfficientNetAggNet


# ─────────────────────────────────────────────
# 4.  WEIGHT INITIALISATION
# ─────────────────────────────────────────────
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


# ─────────────────────────────────────────────
# 5.  LOSS & METRICS
# ─────────────────────────────────────────────
class MonotonicMSELoss(nn.Module):
    """
    MSE + Monotonic Penalty
    ─────────────────────────
    - MSE ปกติ → ทุกตะแกรงสำคัญเท่ากัน
    - บังคับ monotonic: 3/4" >= 1/2" >= 3/8" >= #4 >= #8 >= Pan
    """
    def __init__(self, mono_weight=0.5):
        super().__init__()
        self.mse         = nn.MSELoss()
        self.mono_weight = mono_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mono_diff = pred[:, 1:] - pred[:, :-1]
        penalty   = torch.clamp(mono_diff, min=0).pow(2).mean()
        return mse_loss + self.mono_weight * penalty


class MultiTaskLoss(nn.Module):
    """
    Combined loss for 3 tasks:
      L = L_passing + w_grad * L_gradation + w_prod * L_production
    """
    def __init__(self, mono_weight=0.5, w_grad=0.3, w_prod=0.3):
        super().__init__()
        self.passing_loss = MonotonicMSELoss(mono_weight)
        self.grad_loss    = nn.CrossEntropyLoss()
        self.prod_loss    = nn.MSELoss()
        self.w_grad       = w_grad
        self.w_prod       = w_prod

    def forward(self, passing_pred, target,
                grad_logits=None, grad_target=None,
                prod_pred=None, prod_target=None):
        loss = self.passing_loss(passing_pred, target)
        if grad_logits is not None and grad_target is not None:
            loss = loss + self.w_grad * self.grad_loss(grad_logits, grad_target)
        if prod_pred is not None and prod_target is not None:
            loss = loss + self.w_prod * self.prod_loss(prod_pred, prod_target)
        return loss


def compute_mae(pred, target):
    return (pred - target).abs().mean().item() * 100.0


def compute_metrics(preds, targets, tolerance=10.0):
    """
    คำนวณ metrics ครบ 4 แบบ:
    1. MAE (%) — error เฉลี่ยต่อ sieve-sample cell
    2. Per-sieve Accuracy (%) — % cell ทั้งหมดที่ error <= tolerance
    3. Sample Accuracy (%) — % sample ที่ทุกตะแกรงผ่าน tolerance (QC pass/fail)
    4. R² Score — เฉพาะตะแกรงใหญ่ 3/4", 1/2", 3/8" ที่ variance สูงพอ
    """
    pred_pct   = preds   * 100.0   # shape (N, 6)
    target_pct = targets * 100.0
    per_sieve_err = (pred_pct - target_pct).abs()   # (N, 6)

    # 1. MAE
    mae = per_sieve_err.mean().item()

    # 2. Per-sieve Accuracy — นับ cell ที่ผ่าน / cell ทั้งหมด
    per_sieve_acc = (per_sieve_err <= tolerance).float().mean().item() * 100.0

    # 3. Sample Accuracy — ทุกตะแกรงของ sample ต้องผ่านพร้อมกัน (QC pass/fail)
    sample_correct = (per_sieve_err <= tolerance).all(dim=1)
    sample_acc     = sample_correct.float().mean().item() * 100.0

    # 4. R² เฉพาะตะแกรงใหญ่ (3/4", 1/2", 3/8") ที่มี variance สูงพอ
    R2_SIEVES = [0, 1, 2]
    r2_per_sieve = []
    for s in range(preds.shape[1]):
        p = pred_pct[:, s]
        t = target_pct[:, s]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        if ss_tot < 1e-8:
            r2 = float('nan')
        else:
            r2 = max(-1.0, (1 - ss_res / ss_tot).item())
        r2_per_sieve.append(r2)
    r2_main = [r2_per_sieve[i] for i in R2_SIEVES if not np.isnan(r2_per_sieve[i])]
    r2_avg  = float(np.mean(r2_main)) if r2_main else float('nan')

    return mae, per_sieve_acc, sample_acc, r2_avg, r2_per_sieve


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_mae, total = 0.0, 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, scalars, targets = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            preds = model(imgs, scalars)
            if isinstance(preds, tuple):
                preds = preds[0]
            if isinstance(criterion, MultiTaskLoss):
                loss = criterion.passing_loss(preds, targets)
            else:
                loss = criterion(preds, targets)
            total_loss += loss.item() * imgs.size(0)
            total_mae  += compute_mae(preds, targets) * imgs.size(0)
            total      += imgs.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    avg_loss = total_loss / total
    avg_mae  = total_mae  / total
    return avg_loss, avg_mae, torch.cat(all_preds), torch.cat(all_targets)


# ─────────────────────────────────────────────
# 6.  TRAINING LOOP
# ─────────────────────────────────────────────
def train(agg_filter=None):
    """
    agg_filter: None = train ทุก sample (ไม่แนะนำ)
                '3_4inch' = train เฉพาะ Aggregate 3_4inch
                '3_8inch' = train เฉพาะ Aggregate 3_8inch
    """
    tag = agg_filter.replace('Aggregate ', '').replace('inch', '"') if agg_filter else 'ALL'
    print("\n" + "="*60)
    print(f"  AggNet Single Image  [{tag}]  (ASTM C 136)")
    print("="*60)

    # เลือก sieve columns ตาม agg_filter
    if agg_filter == 'Aggregate 3_8inch':
        sieve_cols   = SIEVE_COLS_38
        sieve_labels = ['#4\n4.75', '#8\n2.36', 'Pan']
        model_name   = "aggnet_38_best.pth"
    elif agg_filter == 'Aggregate 1 inch':
        sieve_cols   = SIEVE_COLS_1INCH
        sieve_labels = SIEVE_LABELS   # 3/4"~Pan (1"≈100% ตัดออกเหมือนกัน)
        model_name   = "aggnet_1inch_best.pth"
    else:
        sieve_cols   = SIEVE_COLS_34
        sieve_labels = SIEVE_LABELS
        model_name   = "aggnet_34_best.pth"
    n_sieves = len(sieve_cols)

    print(f"\n[1] Loading dataset from splits.json ...")

    # Determine model key for splits.json
    if agg_filter == 'Aggregate 3_8inch':
        split_key = 'model_b'
    elif agg_filter == 'Aggregate 1 inch':
        split_key = 'model_c'
    else:
        split_key = 'model_a'

    # Load split IDs
    if not os.path.exists(SPLITS_JSON):
        print(f"[ERROR] splits.json not found: {SPLITS_JSON}")
        return None, None

    splits_data = json.load(open(SPLITS_JSON))
    if split_key not in splits_data:
        print(f"\n  [SKIP] {split_key} not in splits.json → ข้ามการ train")
        return None, None

    train_ids = splits_data[split_key]['train']
    val_ids   = splits_data[split_key]['val']
    test_ids  = splits_data[split_key]['test']

    if len(train_ids) < 3:
        print(f"\n  [SKIP] {agg_filter} train set มีเพียง {len(train_ids)} sample(s) → รอเพิ่มข้อมูล")
        return None, None

    train_rec = load_records(DATA_DIR, LABELS_CSV, sieve_cols=sieve_cols, sample_ids=train_ids)
    val_rec   = load_records(DATA_DIR, LABELS_CSV, sieve_cols=sieve_cols, sample_ids=val_ids)
    test_rec  = load_records(DATA_DIR, LABELS_CSV, sieve_cols=sieve_cols, sample_ids=test_ids)
    print(f"    Train: {len(train_rec)}  |  Val: {len(val_rec)}  |  Test: {len(test_rec)} (held out)")

    train_ds = AggDatasetMT(train_rec, transform=get_transforms(True),  sieve_cols=sieve_cols)
    val_ds   = AggDatasetMT(val_rec,   transform=get_transforms(False), sieve_cols=sieve_cols)
    test_ds  = AggDatasetMT(test_rec,  transform=get_transforms(False), sieve_cols=sieve_cols)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    use_multitask = (n_sieves >= 3 and '3_4inch' in sieve_cols)
    print(f"\n[2] Building EfficientNet-B0 + CBAM + {'Multi-Task' if use_multitask else 'Single-Task'} ...")
    model = EfficientNetAggNet(num_sieves=n_sieves, multitask=use_multitask).to(DEVICE)
    model.head_passing.apply(init_weights)
    model.cbam.apply(init_weights)
    if use_multitask:
        model.head_gradation.apply(init_weights)
        model.head_production.apply(init_weights)

    total_params_all  = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in model.head_passing.parameters())
    cbam_params = sum(p.numel() for p in model.cbam.parameters())
    mt_params   = 0
    if use_multitask:
        mt_params = (sum(p.numel() for p in model.head_gradation.parameters())
                   + sum(p.numel() for p in model.head_production.parameters()))
    print(f"    Total parameters     : {total_params_all:,}")
    print(f"    CBAM parameters      : {cbam_params:,}")
    print(f"    Head (passing)       : {head_params:,}")
    if use_multitask:
        print(f"    Head (grad+prod)     : {mt_params:,}")
    print(f"    Multi-task           : {use_multitask}")
    print(f"    Input : image (batch,3,{IMG_SIZE[0]},{IMG_SIZE[1]}) + [weight_norm, agg_code] (batch,2)")
    print(f"    Output: (batch, {n_sieves})  sieves: {sieve_cols}")

    criterion = MultiTaskLoss(mono_weight=0.5, w_grad=0.3, w_prod=0.3) if use_multitask else MonotonicMSELoss(mono_weight=0.5)
    model_save_path = os.path.join(MODEL_DIR, model_name)

    # ── Phase 1: Freeze backbone, train head only ──
    print(f"\n[3a] Phase 1 — Freeze backbone, train head ({FREEZE_EPOCHS} epochs, LR={LR_HEAD}) ...")
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler_cos     = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=FREEZE_EPOCHS, eta_min=LR_MIN)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                            factor=LR_FACTOR, patience=10, min_lr=LR_MIN)

    best_val_loss = float('inf')
    best_weights  = None
    history       = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'lr': [], 'phase': []}

    def _train_one_epoch(loader, ds_len):
        model.train()
        running_loss = 0.0
        for batch in loader:
            imgs     = batch[0].to(DEVICE)
            scalars  = batch[1].to(DEVICE)
            targets  = batch[2].to(DEVICE)
            grad_cls = batch[3].to(DEVICE)
            prod_rat = batch[4].to(DEVICE)

            optimizer.zero_grad()
            output = model(imgs, scalars)

            if isinstance(output, tuple):
                passing_pred, grad_logits, prod_pred = output
                loss = criterion(passing_pred, targets,
                                 grad_logits, grad_cls,
                                 prod_pred, prod_rat)
            else:
                loss = criterion(output, targets) if not isinstance(criterion, MultiTaskLoss) else criterion.passing_loss(output, targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        return running_loss / ds_len

    for epoch in range(1, FREEZE_EPOCHS + 1):
        train_loss = _train_one_epoch(train_loader, len(train_ds))
        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion)
        scheduler_cos.step()
        scheduler_plateau.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)
        history['phase'].append(1)

        print(f"[P1] Epoch [{epoch:>3}/{FREEZE_EPOCHS}]  "
              f"Train: {train_loss:.5f}  Val: {val_loss:.5f}  MAE: {val_mae:.2f}%  LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            torch.save(best_weights, model_save_path)
            print(f"    ✓ Saved (Val Loss: {best_val_loss:.5f})")

    # ── Phase 2: Unfreeze last 2 blocks, fine-tune ──
    print(f"\n[3b] Phase 2 — Partial unfreeze (last 2 blocks), fine-tune (up to {MAX_EPOCHS} epochs, LR={LR_FINETUNE}) ...")
    model.unfreeze_last_n_blocks(n=2)
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params (Phase 2): {trainable_p2:,}  (last 2 EfficientNet blocks + CBAM + heads)")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler_cos     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                            factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)

    no_improve = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _train_one_epoch(train_loader, len(train_ds))
        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion)
        scheduler_cos.step()
        scheduler_plateau.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)
        history['phase'].append(2)

        print(f"[P2] Epoch [{epoch:>3}/{MAX_EPOCHS}]  "
              f"Train: {train_loss:.5f}  Val: {val_loss:.5f}  MAE: {val_mae:.2f}%  LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            no_improve    = 0
            torch.save(best_weights, model_save_path)
            print(f"    ✓ Saved (Val Loss: {best_val_loss:.5f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {FREEZE_EPOCHS + epoch}")
                break

    model.load_state_dict(best_weights)
    print(f"\n  Best Val Loss: {best_val_loss:.5f}")

    print("\n[4] Final Evaluation ...")
    val_loss, val_mae, val_preds, val_targets = evaluate(
        model, val_loader, criterion)

    # คำนวณ 3 metrics
    mae, per_sieve_acc, sample_acc, r2_avg, r2_per = compute_metrics(val_preds, val_targets, tolerance=10.0)

    print("\n" + "="*65)
    print("  Model Performance Summary")
    print("="*65)
    print(f"  MAE (avg)                    : {mae:.2f}%")
    print(f"  Per-sieve Accuracy  (±10%)   : {per_sieve_acc:.1f}%  ← % cell ที่ error ≤ 10%")
    print(f"  Sample Accuracy     (±10%)   : {sample_acc:.1f}%  ← % sample ที่ผ่านทุกตะแกรง (QC)")
    print(f"  R² Score (3/4\"~3/8\" only)   : {r2_avg:.4f}")
    print("="*65)

    mae_per_sieve = (val_preds - val_targets).abs().mean(dim=0).numpy() * 100
    r2_arr        = np.array(r2_per)

    print("\n  Metrics per sieve:")
    print(f"  {'Sieve':>12}  {'MAE':>7}  {'R²':>7}  {'Per-sieve Acc':>14}  {'Note'}")
    print("  " + "-"*60)
    for i, label in enumerate(sieve_labels):
        lbl      = label.replace('\n', ' ')
        err_s    = (val_preds[:, i] - val_targets[:, i]).abs() * 100
        acc_s    = (err_s <= 10.0).float().mean().item() * 100
        r2_str   = f"{r2_arr[i]:.3f}" if not np.isnan(r2_arr[i]) else "  N/A"
        note     = "" if i < 3 else "(low variance)"
        print(f"  {lbl:>12}  {mae_per_sieve[i]:>6.2f}%  {r2_str:>7}  {acc_s:>13.1f}%  {note}")

    tag_short = tag.replace('/', '').replace('"', '')
    plot_history(history, suffix=tag_short)
    plot_gradation_curves(val_preds, val_targets, val_rec, sieve_labels=sieve_labels,
                          suffix='val')
    plot_metrics_summary(val_preds, val_targets, sieve_labels=sieve_labels,
                         suffix='val')

    # ── Test Set Evaluation (held out) ──
    if len(test_rec) > 0:
        print("\n[5] Test Set Evaluation (held out — never seen during training) ...")
        test_loss, test_mae_avg, test_preds, test_targets = evaluate(
            model, test_loader, criterion)

        t_mae, t_ps_acc, t_s_acc, t_r2, t_r2_per = compute_metrics(
            test_preds, test_targets, tolerance=10.0)

        print("\n" + "="*65)
        print("  TEST SET Performance")
        print("="*65)
        print(f"  MAE (avg)                    : {t_mae:.2f}%")
        print(f"  Per-sieve Accuracy  (±10%)   : {t_ps_acc:.1f}%")
        print(f"  Sample Accuracy     (±10%)   : {t_s_acc:.1f}%")
        print(f"  R² Score (3/4\"~3/8\" only)   : {t_r2:.4f}")
        print("="*65)

        t_mae_arr = (test_preds - test_targets).abs().mean(dim=0).numpy() * 100
        t_r2_arr  = np.array(t_r2_per)
        print(f"\n  {'Sieve':>12}  {'MAE':>7}  {'R²':>7}  {'Acc(±10%)':>10}")
        print("  " + "-"*45)
        for i, label in enumerate(sieve_labels):
            lbl    = label.replace('\n', ' ')
            err_s  = (test_preds[:, i] - test_targets[:, i]).abs() * 100
            acc_s  = (err_s <= 10.0).float().mean().item() * 100
            r2_str = f"{t_r2_arr[i]:.3f}" if not np.isnan(t_r2_arr[i]) else "  N/A"
            print(f"  {lbl:>12}  {t_mae_arr[i]:>6.2f}%  {r2_str:>7}  {acc_s:>9.1f}%")

        plot_gradation_curves(test_preds, test_targets, test_rec,
                              sieve_labels=sieve_labels, suffix='test')
        plot_metrics_summary(test_preds, test_targets, sieve_labels=sieve_labels,
                             suffix='test')

    print(f"\n[6] Model saved: {model_save_path}")
    print("    Done!")
    return model, history


# ─────────────────────────────────────────────
# 7.  VISUALISATION
# ─────────────────────────────────────────────
def plot_metrics_summary(preds, targets, tolerance=10.0, sieve_labels=None, suffix='val'):
    """Plot MAE, R², Tolerance Accuracy แต่ละตะแกรง"""
    mae, per_sieve_acc, sample_acc, r2_avg, r2_per = compute_metrics(preds, targets, tolerance)

    mae_arr = (preds - targets).abs().mean(dim=0).numpy() * 100
    r2_arr  = np.array(r2_per)
    _labels = sieve_labels if sieve_labels is not None else SIEVE_LABELS
    labels  = [l.replace('\n', ' ') for l in _labels]

    # Tolerance accuracy ต่อตะแกรง
    tol_arr = []
    for i in range(preds.shape[1]):
        err = (preds[:, i] - targets[:, i]).abs() * 100
        tol_arr.append((err <= tolerance).float().mean().item() * 100)
    tol_arr = np.array(tol_arr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#e74c3c' if m > 5 else '#2ecc71' for m in mae_arr]

    # MAE per sieve
    axes[0].bar(labels, mae_arr, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].axhline(y=mae, color='navy', linestyle='--', linewidth=1.5,
                    label=f'Avg MAE: {mae:.2f}%')
    axes[0].axhline(y=10.0, color='orange', linestyle=':', linewidth=1.5,
                    label='Tolerance: 10%')
    axes[0].set_title('MAE per Sieve', fontweight='bold')
    axes[0].set_ylabel('MAE (%)')
    axes[0].set_ylim(0, max(mae_arr.max() * 1.3, 8))
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)
    for j, v in enumerate(mae_arr):
        axes[0].text(j, v + 0.2, f'{v:.1f}%', ha='center', fontsize=8)

    # R² per sieve — แสดงเฉพาะ 3 ตะแกรงใหญ่ (3/4", 1/2", 3/8")
    r2_labels_main  = labels[:3]
    r2_vals_main    = r2_arr[:3]
    r2_colors_main  = ['#2ecc71' if r >= 0.8 else '#e74c3c' for r in r2_vals_main]
    axes[1].bar(r2_labels_main, r2_vals_main,
                color=r2_colors_main, edgecolor='white', linewidth=0.5)
    r2_avg_clean = float(np.nanmean(r2_arr[:3]))
    axes[1].axhline(y=r2_avg_clean, color='navy', linestyle='--', linewidth=1.5,
                    label=f'Avg R²: {r2_avg_clean:.4f}')
    axes[1].axhline(y=0.9, color='orange', linestyle=':', linewidth=1.5,
                    label='Target: 0.90')
    axes[1].set_title('R² Score per Sieve\n(3/4", 1/2", 3/8" only)', fontweight='bold')
    axes[1].set_ylabel('R² Score')
    axes[1].set_ylim(min(0, r2_vals_main.min() - 0.1), 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].text(0.98, 0.02, '#4, #8, Pan: MAE-only\n(low variance)',
                 transform=axes[1].transAxes, ha='right', va='bottom',
                 fontsize=7, color='gray',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    for j, v in enumerate(r2_vals_main):
        axes[1].text(j, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)

    # Tolerance Accuracy per sieve — grouped bar (per-sieve vs sample)
    x_idx    = np.arange(len(labels))
    bar_w    = 0.35
    tol_colors_ps = ['#2ecc71' if t >= 80 else '#e74c3c' for t in tol_arr]

    # Per-sieve accuracy
    bars1 = axes[2].bar(x_idx - bar_w/2, tol_arr,
                        width=bar_w, color=tol_colors_ps,
                        edgecolor='white', linewidth=0.5,
                        label=f'Per-sieve Acc: {per_sieve_acc:.1f}%')
    # Sample accuracy (แสดงเป็นเส้นประแทน เพราะเป็นค่าเดียวทั้ง sample)
    axes[2].axhline(y=sample_acc, color='navy', linestyle='--', linewidth=2,
                    label=f'Sample Acc: {sample_acc:.1f}% (QC)')
    axes[2].axhline(y=80, color='orange', linestyle=':', linewidth=1.5,
                    label='Target: 80%')

    axes[2].set_title(f'Per-sieve Accuracy (±{tolerance}%)', fontweight='bold')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_ylim(0, 120)
    axes[2].set_xticks(x_idx - bar_w/2)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].legend(fontsize=7)
    axes[2].grid(axis='y', alpha=0.3)
    for j, v in enumerate(tol_arr):
        axes[2].text(j - bar_w/2, v + 1, f'{v:.0f}%', ha='center', fontsize=7)
    axes[2].text(0.98, sample_acc/120 + 0.02,
                 f'Sample={sample_acc:.0f}%',
                 transform=axes[2].transAxes, ha='right',
                 fontsize=8, color='navy', fontweight='bold')

    r2_display = float(np.nanmean(r2_arr[:3]))
    plt.suptitle(
        f'AggNet Performance Summary  |  MAE: {mae:.2f}%  |  '
        f'R²(3/4"~3/8"): {r2_display:.4f}  |  '
        f'Per-sieve Acc: {per_sieve_acc:.1f}%  |  Sample Acc(QC): {sample_acc:.1f}%',
        fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f'metrics_summary_{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Metrics summary saved: {save_path}")


def plot_history(history, suffix=''):
    has_lr = 'lr' in history and len(history['lr']) > 0
    ncols  = 3 if has_lr else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='steelblue')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='tomato')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history['val_mae'], color='orange', label='Val MAE (%)')
    axes[1].axhline(y=2.0, color='green',  linestyle='--', linewidth=1,
                    label='Target: 2%')
    axes[1].axhline(y=10.0, color='red',    linestyle=':', linewidth=1,
                    label='Tolerance: 10%')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE (%)')
    axes[1].set_title('Validation MAE (% passing)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # LR schedule
    if has_lr:
        axes[2].semilogy(history['lr'], color='purple', label='Learning Rate')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Learning Rate (log)')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f'history_{suffix}.png' if suffix else 'single_history.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    History saved: {save_path}")


def plot_gradation_curves(preds, targets, records, sieve_labels=None, suffix='val'):
    n    = len(preds)
    n_sieves = preds.shape[1]
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).reshape(rows, cols)

    x_pos = list(range(n_sieves))

    for i in range(n):
        pred_pct   = preds[i].numpy()   * 100
        target_pct = targets[i].numpy() * 100
        r          = records[i]
        ax         = axes[i // cols][i % cols]

        ax.plot(x_pos, target_pct, 'b-o', linewidth=2,
                markersize=6, label='Actual (Lab)')
        ax.plot(x_pos, pred_pct,   'r--s', linewidth=2,
                markersize=6, label='Predicted')
        _sl = sieve_labels if sieve_labels is not None else SIEVE_LABELS
        ax.set_xticks(x_pos)
        ax.set_xticklabels(_sl, fontsize=7)
        ax.set_ylim(-5, 110)
        ax.set_xlabel('Sieve Size')
        ax.set_ylabel('% Passing')
        mae   = np.abs(pred_pct - target_pct).mean()
        title = f"Sample {r['sample_id']}  |  MAE: {mae:.1f}%"
        if r['source']:
            title += f"\n{r['source'][:35]}"
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    set_label = 'Validation' if suffix == 'val' else 'Test (held out)'
    plt.suptitle(f'AggNet: Predicted vs Actual Gradation Curves ({set_label})',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f'gradation_{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Gradation curves saved: {save_path}")


# ─────────────────────────────────────────────
# 8.  INFERENCE
# ─────────────────────────────────────────────
def predict(image_path, weight_g=600.0, agg_type='Aggregate 3_4inch',
            model_path=None, plot=True, source_name=""):
    """
    รับภาพรวมหิน 1 ภาพ + น้ำหนักตัวอย่าง + ประเภทหิน → คืน gradation curve

    Args:
        image_path  : path ไปยังภาพรวมหินก่อนร่อน (jpg/png/...)
        weight_g    : น้ำหนักตัวอย่างที่ถ่ายภาพ (กรัม) default=600g
        agg_type    : 'Aggregate 3_4inch' หรือ 'Aggregate 3_8inch'
        model_path  : path ไฟล์ .pth
        plot        : แสดงกราฟหรือไม่
        source_name : ชื่อแหล่งหิน

    Returns:
        dict { sieve_label: % passing }
    """
    if model_path is None:
        model_path = os.path.join(SAVE_DIR, "aggnet_34_best.pth")

    model = EfficientNetAggNet(num_sieves=NUM_SIEVES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    tf  = get_transforms(train=False)
    img = Image.open(image_path).convert("RGB")
    x   = tf(img).unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

    weight_norm = float(np.clip(
        (weight_g - WEIGHT_MIN) / (WEIGHT_MAX - WEIGHT_MIN), 0.0, 1.0))
    agg_code = AGG_TYPE_MAP.get(agg_type, 0.0)
    w = torch.tensor([[weight_norm, agg_code]], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = model(x, w).squeeze().cpu().numpy() * 100   # → %

    # Print results
    print("\n" + "="*55)
    print("  Predicted Gradation Curve  (ASTM C 136 – 7 Sieves)")
    print("="*55)
    print(f"  {'Sieve':>12}  {'Size (mm)':>10}  {'% Passing':>10}")
    print("-"*55)
    for label, size, pct in zip(SIEVE_LABELS, SIEVE_SIZES, pred):
        lbl      = label.replace('\n', ' ')
        size_str = f"{size:.2f}" if size > 0.01 else "Pan"
        print(f"  {lbl:>12}  {size_str:>10}  {pct:>9.1f}%")

    if plot:
        x_pos = list(range(NUM_SIEVES))
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x_pos, pred, 'r-o', linewidth=2.5, markersize=8,
                label='Predicted', zorder=3)
        ax.fill_between(x_pos, 0, pred, alpha=0.08, color='red')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(SIEVE_LABELS, fontsize=9)
        ax.set_ylim(-5, 110)
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0,   color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Sieve Size', fontsize=12)
        ax.set_ylabel('% Passing', fontsize=12)
        title = f'Predicted Gradation Curve\n{source_name or Path(image_path).name}'
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, 'single_predicted.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n  Plot saved: {save_path}")

    return dict(zip(SIEVE_LABELS, pred.tolist()))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Train แยกตาม Aggregate Type ──
    print("\n" + "█"*60)
    print("  MODEL A: Aggregate 3/4\" (6 sieves)")
    print("█"*60)
    model_34, history_34 = train(agg_filter='Aggregate 3_4inch')

    print("\n" + "█"*60)
    print("  MODEL B: Aggregate 3/8\" (3 sieves: No4/No8/Pan)")
    print("  หมายเหตุ: 3/4\"=0, 1/2\"=100, 3/8\"≈99.7 → constant ไม่ต้อง predict")
    print("█"*60)
    model_38, history_38 = train(agg_filter='Aggregate 3_8inch')

    print("\n" + "█"*60)
    print("  MODEL C: Aggregate 1\" (6 sieves: 3/4\"~Pan)")
    print("  หมายเหตุ: 1\"≈100% constant ไม่ต้อง predict — skip อัตโนมัติถ้า sample < 4")
    print("█"*60)
    model_1inch, history_1inch = train(agg_filter='Aggregate 1 inch')

    # ── Inference (uncomment แล้วใส่ path ภาพ) ──
    # result = predict(
    #     image_path  = "/workspace/AggNet/data/dataset3/Sample_006.jpg",
    #     weight_g    = 620,
    #     agg_type    = "Aggregate 3_4inch",   # หรือ "Aggregate 3_8inch"
    #     source_name = "ART CONCRETE COMPANY LIMITED"
    # )
    # print(result)