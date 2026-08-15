"""
Validate data/dataset3/labels.csv - run before adding new data to training set.
Checks: missing values, duplicate sample_id, weight range, %Passing range,
monotonicity, Aggregate Type consistency, missing images, label duplicates.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = "data/dataset3"
LABELS_CSV = f"{DATA_DIR}/labels.csv"
SIEVE_COLS = ['1inch', '3_4inch', '1_2inch', '3_8inch', 'No4', 'No8', 'Pan']
VALID_AGG_TYPES = {'Aggregate 3_4inch', 'Aggregate 3_8inch', 'Aggregate 1 inch'}
WEIGHT_MIN, WEIGHT_MAX = 400, 900

out = []
def log(msg=""):
    out.append(str(msg))

df_raw = pd.read_csv(LABELS_CSV)
df = df_raw.copy()
df.columns = df.columns.str.strip()

log(f"=== BASIC INFO ===")
log(f"Total rows: {len(df)}")
log(f"Columns: {list(df.columns)}")
log()

# 1. Missing values
log("=== 1. MISSING VALUES ===")
na_counts = df.isnull().sum()
na_counts = na_counts[na_counts > 0]
if len(na_counts) == 0:
    log("None found.")
else:
    log(na_counts.to_string())
log()

# 2. Duplicate sample_id
log("=== 2. DUPLICATE sample_id ===")
dup_ids = df[df.duplicated(subset=['sample_id'], keep=False)].sort_values('sample_id')
if dup_ids.empty:
    log("None found.")
else:
    log(f"{dup_ids['sample_id'].nunique()} sample_id(s) appear more than once:")
    log(dup_ids[['sample_id','Aggregate Type','Source']].to_string())
log()

# 3. Weight range
log(f"=== 3. WEIGHT OUT OF RANGE ({WEIGHT_MIN}-{WEIGHT_MAX}g) ===")
bad_weight = df[(df['weight_g'] < WEIGHT_MIN) | (df['weight_g'] > WEIGHT_MAX)]
if bad_weight.empty:
    log("None found. weight_g range in data:", )
    log(f"  min={df['weight_g'].min()}  max={df['weight_g'].max()}")
else:
    log(f"{len(bad_weight)} row(s) out of range:")
    log(bad_weight[['sample_id','weight_g','Aggregate Type']].to_string())
log()

# 4. %Passing out of 0-100
log("=== 4. %PASSING OUT OF [0,100] RANGE ===")
found_any = False
for col in SIEVE_COLS:
    bad = df[(df[col] < 0) | (df[col] > 100)]
    if not bad.empty:
        found_any = True
        log(f"Column '{col}': {len(bad)} bad row(s)")
        log(bad[['sample_id', col]].to_string())
if not found_any:
    log("None found.")
log()

# 5. Monotonicity check (should be non-increasing left to right)
log("=== 5. MONOTONICITY VIOLATIONS (sieve % should be non-increasing) ===")
mono_bad = []
for idx, row in df.iterrows():
    vals = [row[c] for c in SIEVE_COLS]
    for i in range(len(vals) - 1):
        if vals[i] < vals[i+1] - 0.01:  # small tolerance for float rounding
            mono_bad.append((row['sample_id'], SIEVE_COLS[i], vals[i], SIEVE_COLS[i+1], vals[i+1]))
if not mono_bad:
    log("None found.")
else:
    log(f"{len(set(x[0] for x in mono_bad))} sample(s) with violations:")
    for sid, c1, v1, c2, v2 in mono_bad:
        log(f"  sample_id={sid}: {c1}={v1} < {c2}={v2}")
log()

# 6. Aggregate Type consistency
log("=== 6. AGGREGATE TYPE VALUES ===")
agg_counts = df['Aggregate Type'].astype(str).str.strip().value_counts()
log(agg_counts.to_string())
unexpected = set(agg_counts.index) - VALID_AGG_TYPES
if unexpected:
    log(f"\n[WARNING] Unexpected Aggregate Type values (typo/spacing?): {unexpected}")
else:
    log("\nAll values match expected 3 categories.")
log()

# 7. Source uniqueness
log("=== 7. SOURCE FIELD ===")
src = df['Source'].astype(str).str.strip()
log(f"Unique Source strings: {src.nunique()}")
log()

# 8. Missing images
log("=== 8. MISSING IMAGE FILES ===")
missing_imgs = []
for sid in df['sample_id']:
    found = False
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        p = Path(DATA_DIR) / f"Sample_{int(sid):03d}{ext}"
        if p.exists():
            found = True
            break
    if not found:
        missing_imgs.append(sid)
if not missing_imgs:
    log("None found - every sample_id has a matching image.")
else:
    log(f"{len(missing_imgs)} sample_id(s) missing image files: {missing_imgs}")
log()

# 9. Orphan images (image exists but no CSV row)
log("=== 9. ORPHAN IMAGES (image file with no labels.csv row) ===")
csv_ids = set(df['sample_id'].astype(int))
img_ids = set()
for f in Path(DATA_DIR).glob("Sample_*.jpg"):
    try:
        img_ids.add(int(f.stem.replace("Sample_", "")))
    except ValueError:
        pass
orphans = sorted(img_ids - csv_ids)
if not orphans:
    log("None found.")
else:
    log(f"{len(orphans)} orphan image(s): {orphans}")
log()

# 10. Label-duplicate detection (same logic as clean_duplicates() in aggnet_dataset3.py)
log("=== 10. LABEL-DUPLICATE SAMPLES (same %Passing within same Aggregate Type) ===")
sieve_check_cols = ['3_4inch', '1_2inch', '3_8inch', 'No4', 'No8', 'Pan']
dup_total = 0
for agg_type, grp in df.groupby('Aggregate Type'):
    dup_mask = grp.duplicated(subset=sieve_check_cols, keep='first')
    dup_rows = grp[dup_mask]
    if not dup_rows.empty:
        dup_total += len(dup_rows)
        for _, row in dup_rows.iterrows():
            log(f"  Sample_{int(row['sample_id']):03d} ({agg_type}) has duplicate label")
if dup_total == 0:
    log("None found.")
else:
    log(f"\nTotal: {dup_total} sample(s) will be auto-dropped by clean_duplicates()")
log()

# 11. Date parseability
log("=== 11. TESTED DATE FORMAT ===")
bad_dates = []
for idx, row in df.iterrows():
    d = str(row['Tested Date']).strip()
    try:
        pd.to_datetime(d, dayfirst=True, errors='raise')
    except Exception:
        bad_dates.append((row['sample_id'], d))
if not bad_dates:
    log("All dates parseable.")
else:
    log(f"{len(bad_dates)} unparseable date(s):")
    for sid, d in bad_dates:
        log(f"  sample_id={sid}: '{d}'")
log()

# 12. Per-type counts (post dedup)
log("=== 12. SAMPLE COUNTS PER TYPE (raw vs after label-dedup) ===")
df_clean = df.copy()
keep_mask = pd.Series(True, index=df.index)
for agg_type, grp in df.groupby('Aggregate Type'):
    dup_mask = grp.duplicated(subset=sieve_check_cols, keep='first')
    keep_mask.loc[grp[dup_mask].index] = False
df_clean = df[keep_mask]
for t in sorted(VALID_AGG_TYPES):
    raw_n = (df['Aggregate Type'].astype(str).str.strip() == t).sum()
    clean_n = (df_clean['Aggregate Type'].astype(str).str.strip() == t).sum()
    log(f"  {t}: raw={raw_n}  after_dedup={clean_n}")
log(f"  TOTAL: raw={len(df)}  after_dedup={len(df_clean)}")

result = "\n".join(out)
print(result)
with open("validation_report.txt", "w", encoding="utf-8") as f:
    f.write(result)
