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

labels.csv format:
    sample_id, 3_4inch, 1_2inch, 3_8inch, No4, No8, Pan, Aggregate Type, Source, Tested Date
    1, 88.6, 34.8, 11.1, 0.2, 0.2, 0.0, Aggregate 3_4inch, ART CONCRETE, 2025-09-30
    2, 85.2, 16.9, 6.0,  0.3, 0.3, 0.0, Aggregate 3_4inch, BNT CONCRETE, 2025-10-01
    ...

หมายเหตุ:
    - sample_id=1 → ภาพชื่อ Sample_001.jpg (รองรับ .jpg .jpeg .png .bmp .tiff)
    - ตัด 1inch ออก เพราะ % passing = 100% ทุก sample (ไม่มีประโยชน์ในการเรียนรู้)
    - ค่า % Passing ช่วง 0.0 – 100.0
"""

import os
import copy
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
from torchvision import transforms
from PIL import Image

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR   = "/workspace/AggNet/data/dataset_single"
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
SAVE_DIR   = "/workspace/AggNet/outputs/single"
IMG_SIZE   = (224, 224)   # H x W
BATCH_SIZE   = 4          # ปรับตาม dataset size
LR           = 0.0003     # ลดจาก 0.0005 → stable training
MAX_EPOCHS   = 300
PATIENCE     = 40         # เพิ่มขึ้นเพื่อรอ LR schedule
VAL_SPLIT    = 0.33
SEED         = 42

# ── LR Scheduler config ──
LR_FACTOR    = 0.5        # ลด LR เหลือ 50% เมื่อ val loss ไม่ดีขึ้น
LR_PATIENCE  = 10         # รอ 10 epoch ก่อนลด LR
LR_MIN       = 1e-6       # LR ต่ำสุด
WEIGHT_DECAY = 5e-4       # เพิ่มจาก 1e-4 → ลด overfit

# ── ASTM C 136 Sieve config (6 ตะแกรง ตัด 1inch ออก) ──
SIEVE_COLS   = ['3_4inch', '1_2inch', '3_8inch', 'No4', 'No8', 'Pan']
SIEVE_SIZES  = [19.00, 12.50, 9.50, 4.75, 2.36, 0.001]
SIEVE_LABELS = ['3/4"\n19.0', '1/2"\n12.5', '3/8"\n9.50',
                '#4\n4.75',   '#8\n2.36',   'Pan']
NUM_SIEVES   = len(SIEVE_COLS)   # 6

# ── ชื่อภาพ format ──
# sample_id=1 → Sample_001.jpg
IMG_PREFIX = "Sample"

os.makedirs(SAVE_DIR, exist_ok=True)
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
        label    = rec['label']   # shape (6,) ค่า 0.0–1.0

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


def find_image(data_dir, sample_id):
    """หาไฟล์ภาพของ sample_id รองรับหลาย extension"""
    base = Path(data_dir) / f"{IMG_PREFIX}_{sample_id:03d}"
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        p = Path(str(base) + ext)
        if p.exists():
            return p
    return None


def load_records(data_dir, labels_csv):
    df = pd.read_csv(labels_csv)
    df.columns = df.columns.str.strip()

    records = []
    missing = []

    for _, row in df.iterrows():
        sid      = int(row['sample_id'])
        img_path = find_image(data_dir, sid)

        if img_path is None:
            missing.append(f"Sample_{sid:03d}.*")
            continue

        pct   = np.array([float(row[c]) for c in SIEVE_COLS], dtype=np.float32)
        label = pct / 100.0

        records.append({
            'sample_id': sid,
            'img_path':  img_path,
            'label':     label,
            'source':    str(row.get('Source', '')),
            'agg_type':  str(row.get('Aggregate Type', '')),
        })

    if missing:
        print(f"[WARNING] Images not found: {missing}")

    print(f"  Loaded {len(records)} samples from {labels_csv}")
    for r in records:
        pct_str = ', '.join(f"{v*100:.1f}" for v in r['label'])
        print(f"    Sample_{r['sample_id']:03d}  [{pct_str}]  ← {r['img_path'].name}")

    return records


def split_train_val(records, val_split=0.33, seed=42):
    rng  = random.Random(seed)
    data = records.copy()
    rng.shuffle(data)
    n_val   = max(1, int(len(data) * val_split))
    val_rec = data[:n_val]
    trn_rec = data[n_val:]
    return trn_rec, val_rec


# ─────────────────────────────────────────────
# 2.  TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms(train=True):
    if train:
        # Heavy augmentation — ชดเชย dataset ที่มีน้อย
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────
# 3.  MODEL  (AggNet Single Image)
# ─────────────────────────────────────────────
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.dw   = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                              padding=padding, groups=in_ch, bias=False)
        self.pw   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(x))))


class msEncModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        mid_ch = max(out_ch // 3, 1)
        self.d1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.d2 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.d4 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.dw_sep     = DepthwiseSeparableConv(mid_ch * 3, out_ch)
        self.merge_bn   = nn.BatchNorm2d(out_ch)
        self.merge_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        ms  = torch.cat([self.d1(x), self.d2(x), self.d4(x)], dim=1)
        ms  = self.dw_sep(ms)
        ms  = F.adaptive_avg_pool2d(ms, (res.shape[2], res.shape[3]))
        return self.merge_relu(self.merge_bn(res + ms))


class AggNet(nn.Module):
    """
    AggNet Single Image Regression
    ────────────────────────────────
    Input  : (batch, 3, H, W)   — 1 ภาพรวมหิน RGB
    Process: Stem → msEnc × 3 → Dropout → GAP → Sigmoid
    Output : (batch, 6)          — % passing 6 ตะแกรง (ค่า 0–1)

    Model เล็ก (32→64→128) เพื่อป้องกัน overfit บน dataset ขนาดเล็ก
    """
    def __init__(self, num_sieves=NUM_SIEVES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.enc1   = msEncModule(32,  64)
        self.enc2   = msEncModule(64,  128)
        self.enc3   = msEncModule(128, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.head    = nn.Conv2d(128, num_sieves, 1)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.stem(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.dropout(x)
        x = self.head(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.sigmoid(x)   # (batch, 6) ค่า 0–1


# ─────────────────────────────────────────────
# 4.  WEIGHT INITIALISATION
# ─────────────────────────────────────────────
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────
# 5.  LOSS & METRICS
# ─────────────────────────────────────────────
class MonotonicMSELoss(nn.Module):
    """
    MSE + Monotonic Penalty
    ─────────────────────────
    - MSE ปกติ (ไม่มี weight พิเศษ) → ทุกตะแกรงสำคัญเท่ากัน
    - บังคับ monotonic: 3/4" >= 1/2" >= 3/8" >= #4 >= #8 >= Pan
    - ดีกว่า Weighted Loss สำหรับ dataset ขนาดเล็ก
      เพราะไม่บิดเบือน gradient ของตะแกรงใดตะแกรงหนึ่งมากเกินไป
    """
    def __init__(self, mono_weight=0.5):
        super().__init__()
        self.mse         = nn.MSELoss()
        self.mono_weight = mono_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)

        # Monotonic penalty: pred[:,i] ควรมากกว่า pred[:,i+1]
        mono_diff = pred[:, 1:] - pred[:, :-1]       # ควรเป็น <= 0
        penalty   = torch.clamp(mono_diff, min=0).pow(2).mean()

        return mse_loss + self.mono_weight * penalty


def compute_mae(pred, target):
    return (pred - target).abs().mean().item() * 100.0


def compute_metrics(preds, targets, tolerance=5.0):
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
        for imgs, targets in loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            preds      = model(imgs)
            loss       = criterion(preds, targets)
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
def train():
    print("\n" + "="*60)
    print("  AggNet Single Image  (6 Sieves ASTM C 136)")
    print("="*60)

    print(f"\n[1] Loading dataset ...")
    all_records = load_records(DATA_DIR, LABELS_CSV)
    if len(all_records) == 0:
        print("[ERROR] No samples found. Check DATA_DIR and labels.csv")
        return None, None

    train_rec, val_rec = split_train_val(all_records, VAL_SPLIT, SEED)
    print(f"    Train: {len(train_rec)}  |  Val: {len(val_rec)}")

    train_ds = AggDataset(train_rec, transform=get_transforms(True))
    val_ds   = AggDataset(val_rec,   transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"\n[2] Building AggNet (Single Image) ...")
    model = AggNet(num_sieves=NUM_SIEVES).to(DEVICE)
    model.apply(init_weights)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable parameters: {total_params:,}")
    print(f"    Input : (batch, 3, {IMG_SIZE[0]}, {IMG_SIZE[1]})  [1 img RGB]")
    print(f"    Output: (batch, {NUM_SIEVES})  [% passing 0–1]")

    criterion = MonotonicMSELoss(mono_weight=0.5)

    # AdamW ดีกว่า Adam สำหรับ regularization
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingWarmRestarts — ลด LR แบบ cosine แล้ว restart
    # ช่วยหลุดจาก local minima ได้ดีกว่า ReduceLROnPlateau
    scheduler_cos = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=LR_MIN)

    # ReduceLROnPlateau — สำรองไว้ใช้ถ้า val loss ไม่ดีขึ้น
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN)

    print(f"    Optimizer : AdamW  (lr={LR}, weight_decay={WEIGHT_DECAY})")
    print(f"    Scheduler : CosineAnnealingWarmRestarts (T0=30, Tmult=2)")
    print(f"    Loss      : MonotonicMSELoss (mono_weight=0.5)")

    print("\n[3] Training ...")
    best_val_loss = float('inf')
    best_weights  = None
    no_improve    = 0
    history       = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'lr': []}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion)
        scheduler_cos.step()              # step ทุก epoch
        scheduler_plateau.step(val_loss)  # step เมื่อ val loss ไม่ดีขึ้น

        # log LR ปัจจุบัน
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)

        print(f"Epoch [{epoch:>3}/{MAX_EPOCHS}]  "
              f"Train Loss: {train_loss:.5f}  "
              f"Val Loss: {val_loss:.5f}  "
              f"Val MAE: {val_mae:.2f}%  "
              f"LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            no_improve    = 0
            torch.save(best_weights,
                       os.path.join(SAVE_DIR, "aggnet_single_best.pth"))
            print(f"    ✓ Saved best model (Val Loss: {best_val_loss:.5f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    print(f"\n  Best Val Loss: {best_val_loss:.5f}")

    print("\n[4] Final Evaluation ...")
    val_loss, val_mae, val_preds, val_targets = evaluate(
        model, val_loader, criterion)

    # คำนวณ 3 metrics
    mae, per_sieve_acc, sample_acc, r2_avg, r2_per = compute_metrics(val_preds, val_targets, tolerance=5.0)

    print("\n" + "="*65)
    print("  Model Performance Summary")
    print("="*65)
    print(f"  MAE (avg)                    : {mae:.2f}%")
    print(f"  Per-sieve Accuracy  (±5%)    : {per_sieve_acc:.1f}%  ← % cell ที่ error ≤ 5%")
    print(f"  Sample Accuracy     (±5%)    : {sample_acc:.1f}%  ← % sample ที่ผ่านทุกตะแกรง (QC)")
    print(f"  R² Score (3/4\"~3/8\" only)   : {r2_avg:.4f}")
    print("="*65)

    mae_per_sieve = (val_preds - val_targets).abs().mean(dim=0).numpy() * 100
    r2_arr        = np.array(r2_per)

    print("\n  Metrics per sieve:")
    print(f"  {'Sieve':>12}  {'MAE':>7}  {'R²':>7}  {'Per-sieve Acc':>14}  {'Note'}")
    print("  " + "-"*60)
    for i, (label, size) in enumerate(zip(SIEVE_LABELS, SIEVE_SIZES)):
        lbl      = label.replace('\n', ' ')
        err_s    = (val_preds[:, i] - val_targets[:, i]).abs() * 100
        acc_s    = (err_s <= 5.0).float().mean().item() * 100
        r2_str   = f"{r2_arr[i]:.3f}" if not np.isnan(r2_arr[i]) else "  N/A"
        note     = "" if i < 3 else "(low variance)"
        print(f"  {lbl:>12}  {mae_per_sieve[i]:>6.2f}%  {r2_str:>7}  {acc_s:>13.1f}%  {note}")

    plot_history(history)
    plot_gradation_curves(val_preds, val_targets, val_rec)
    plot_metrics_summary(val_preds, val_targets)

    save_path = os.path.join(SAVE_DIR, "aggnet_single_best.pth")
    print(f"\n[5] Model saved: {save_path}")
    print("    Done!")
    return model, history


# ─────────────────────────────────────────────
# 7.  VISUALISATION
# ─────────────────────────────────────────────
def plot_metrics_summary(preds, targets, tolerance=5.0):
    """Plot MAE, R², Tolerance Accuracy แต่ละตะแกรง"""
    mae, per_sieve_acc, sample_acc, r2_avg, r2_per = compute_metrics(preds, targets, tolerance)

    mae_arr = (preds - targets).abs().mean(dim=0).numpy() * 100
    r2_arr  = np.array(r2_per)
    labels  = [l.replace('\n', ' ') for l in SIEVE_LABELS]

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
    axes[0].axhline(y=5.0, color='orange', linestyle=':', linewidth=1.5,
                    label='Tolerance: 5%')
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
    save_path = os.path.join(SAVE_DIR, 'metrics_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Metrics summary saved: {save_path}")


def plot_history(history):
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
    axes[1].axhline(y=5.0, color='red',    linestyle=':', linewidth=1,
                    label='Tolerance: 5%')
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
    save_path = os.path.join(SAVE_DIR, 'single_history.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    History saved: {save_path}")


def plot_gradation_curves(preds, targets, records):
    n    = len(preds)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).reshape(rows, cols)

    x_pos = list(range(NUM_SIEVES))

    for i in range(n):
        pred_pct   = preds[i].numpy()   * 100
        target_pct = targets[i].numpy() * 100
        r          = records[i]
        ax         = axes[i // cols][i % cols]

        ax.plot(x_pos, target_pct, 'b-o', linewidth=2,
                markersize=6, label='Actual (Lab)')
        ax.plot(x_pos, pred_pct,   'r--s', linewidth=2,
                markersize=6, label='Predicted')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(SIEVE_LABELS, fontsize=7)
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

    plt.suptitle('AggNet: Predicted vs Actual Gradation Curves (Validation)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'single_gradation_val.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Gradation curves saved: {save_path}")


# ─────────────────────────────────────────────
# 8.  INFERENCE
# ─────────────────────────────────────────────
def predict(image_path, model_path=None, plot=True, source_name=""):
    """
    รับภาพรวมหิน 1 ภาพ → คืน gradation curve

    Args:
        image_path  : path ไปยังภาพรวมหินก่อนร่อน (jpg/png/...)
        model_path  : path ไฟล์ .pth (default: SAVE_DIR/aggnet_single_best.pth)
        plot        : แสดงกราฟหรือไม่
        source_name : ชื่อแหล่งหิน (สำหรับ title กราฟ)

    Returns:
        dict { sieve_label: % passing }
    """
    if model_path is None:
        model_path = os.path.join(SAVE_DIR, "aggnet_single_best.pth")

    model = AggNet(num_sieves=NUM_SIEVES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    tf  = get_transforms(train=False)
    img = Image.open(image_path).convert("RGB")
    x   = tf(img).unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy() * 100   # → %

    # Print results
    print("\n" + "="*55)
    print("  Predicted Gradation Curve  (ASTM C 136 – 6 Sieves)")
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
    # ── Train ──
    model, history = train()

    # ── Inference (uncomment แล้วใส่ path ภาพ) ──
    # result = predict(
    #     image_path  = "/workspace/AggNet/data/dataset_single/Sample_006.jpg",
    #     source_name = "ART CONCRETE COMPANY LIMITED"
    # )
    # print(result)