"""
export_labels_from_db.py
========================
ดึง record ที่มี label_json (กรอกจาก Lab Label modal) ออกมาจาก DB
แล้วเพิ่มเข้า labels.csv + copy รูปภาพไปยัง dataset folder

วิธีใช้ (บน RAILAB):
    cd /workspace/AggNet
    python code/export_labels_from_db.py

ผลลัพธ์:
    data/dataset3/labels.csv     ← append record ใหม่
    data/dataset3/Sample_XXX.jpg ← copy จาก uploads/
"""

import os, sys, json, shutil, csv
from pathlib import Path

# ── path ───────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent          # /workspace/AggNet
UPLOAD_DIR  = BASE_DIR / "uploads"
DATA_DIR    = BASE_DIR / "data" / "dataset3"
LABELS_CSV  = DATA_DIR / "labels.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR / "code"))
import db as aggnet_db

# ── อ่าน sample_id ล่าสุดใน labels.csv ────────────────────────────────────────
def get_last_sample_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    return max(int(r['sample_id']) for r in rows)

# ── ดึงข้อมูลจาก DB ────────────────────────────────────────────────────────────
def fetch_labeled_records():
    """ดึง analyses ที่มี label_json ทุก company"""
    with aggnet_db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, company_id, doc_no, agg_type, source,
                       tested_date, label_json
                FROM analyses
                WHERE label_json IS NOT NULL
                ORDER BY company_id, tested_date, id
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

# ── แปลง label_json → ค่า passing ────────────────────────────────────────────
SIEVE_ORDER = ['1inch', '3_4inch', '1_2inch', '3_8inch', 'No4', 'No8', 'Pan']
SIEVE_LABEL_MAP = {
    '1"':   '1inch',
    '3/4"': '3_4inch',
    '1/2"': '1_2inch',
    '3/8"': '3_8inch',
    '#4':   'No4',
    '#8':   'No8',
    'Pan':  'Pan',
}

def parse_label_json(label_json) -> dict | None:
    """คืน dict {1inch:.., 3_4inch:.., ...} หรือ None ถ้าข้อมูลไม่ครบ"""
    if isinstance(label_json, str):
        label_json = json.loads(label_json)
    sieves = label_json.get('sieves', [])
    result = {}
    for s in sieves:
        key = SIEVE_LABEL_MAP.get(s.get('sieve', ''))
        if key and s.get('passing') is not None:
            result[key] = round(float(s['passing']), 2)
    # ต้องมีอย่างน้อย 3_4inch หรือ No4 (แล้วแต่ประเภทหิน)
    if len(result) < 3:
        return None
    return result

# ── หา path รูปภาพ ─────────────────────────────────────────────────────────────
def find_image(company_id, doc_no) -> Path | None:
    folder = UPLOAD_DIR / str(company_id)
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        p = folder / f"{doc_no}{ext}"
        if p.exists():
            return p
    return None

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    records = fetch_labeled_records()
    print(f"พบ {len(records)} record ที่มี label_json ใน DB")

    # อ่าน doc_no ที่มีอยู่ใน CSV แล้ว (ไม่ต้อง import ซ้ำ)
    existing_doc_nos = set()
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row.get('doc_no'):
                    existing_doc_nos.add(row['doc_no'])
    print(f"มีใน CSV แล้ว: {len(existing_doc_nos)} record")

    last_id = get_last_sample_id(LABELS_CSV)
    new_rows = []
    skipped  = []

    for rec in records:
        doc_no = rec.get('doc_no') or ''
        if doc_no in existing_doc_nos:
            skipped.append(doc_no)
            continue

        label = parse_label_json(rec['label_json'])
        if label is None:
            print(f"  ⚠️  {doc_no}: label_json ไม่ครบ ข้าม")
            continue

        img_path = find_image(rec['company_id'], doc_no)
        if img_path is None:
            print(f"  ⚠️  {doc_no}: ไม่พบรูปภาพ ข้าม")
            continue

        # copy รูปไปยัง dataset folder
        last_id += 1
        dest_name = f"Sample_{last_id:03d}{img_path.suffix}"
        shutil.copy2(img_path, DATA_DIR / dest_name)

        # สร้าง row CSV
        tested_date = str(rec.get('tested_date') or '')
        source      = rec.get('source') or ''
        agg_type    = rec.get('agg_type') or 'Aggregate 3_4inch'
        weight_g    = rec['label_json'].get('weight_total_g', '') if isinstance(rec['label_json'], dict) else \
                      json.loads(rec['label_json']).get('weight_total_g', '')

        row = {
            'sample_id':      last_id,
            'doc_no':         doc_no,
            '1inch':          label.get('1inch', 100.0),
            '3_4inch':        label.get('3_4inch', ''),
            '1_2inch':        label.get('1_2inch', ''),
            '3_8inch':        label.get('3_8inch', ''),
            'No4':            label.get('No4', ''),
            'No8':            label.get('No8', ''),
            'Pan':            label.get('Pan', ''),
            'weight_g':       weight_g,
            'Aggregate Type': agg_type,
            'Source':         source,
            'Tested Date':    tested_date,
        }
        new_rows.append(row)
        existing_doc_nos.add(doc_no)
        print(f"  ✅ {doc_no} → {dest_name}  ({agg_type})")

    if not new_rows:
        print("ไม่มี record ใหม่ที่ต้อง export")
        return

    # append เข้า CSV (สร้างใหม่ถ้ายังไม่มี)
    fieldnames = ['sample_id','doc_no','1inch','3_4inch','1_2inch','3_8inch',
                  'No4','No8','Pan','weight_g','Aggregate Type','Source','Tested Date']
    write_header = not LABELS_CSV.exists()
    with open(LABELS_CSV, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)

    print(f"\nเพิ่ม {len(new_rows)} record ใหม่เข้า {LABELS_CSV}")
    if skipped:
        print(f"ข้าม {len(skipped)} record (มีแล้ว): {skipped[:5]}{'...' if len(skipped)>5 else ''}")

if __name__ == '__main__':
    main()
