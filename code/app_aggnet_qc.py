"""
AggNet QC Web App
=================
Flask web application สำหรับ QC Check หิน
- อัพโหลดภาพจากมือถือ
- วิเคราะห์ด้วย AggNet Single Image Model
- Generate PDF Report (SIEVE ANALYSIS AGGREGATE format)

Requirements:
    pip install flask reportlab matplotlib numpy torch torchvision pillow pandas

Usage:
    python3 app_aggnet_qc.py
    เปิด browser: http://10.100.16.22:5000
"""

import os
import io
import sys
import uuid
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string, make_response, session, redirect, url_for

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer, Image as RLImage, HRFlowable, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os as _os
_FONT_PATH = _os.path.join(_os.path.dirname(__file__), 'Sarabun-Regular.ttf')
if _os.path.exists(_FONT_PATH):
    pdfmetrics.registerFont(TTFont('Sarabun', _FONT_PATH))
    _THAI_FONT = 'Sarabun'
else:
    _THAI_FONT = 'Helvetica'

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH_34 = "/workspace/AggNet/models/dataset4/aggnet_34_best.pth"   # Aggregate 3_4inch (6 sieves)
MODEL_PATH_38 = "/workspace/AggNet/models/dataset4/aggnet_38_best.pth"   # Aggregate 3_8inch (3 sieves: No4/No8/Pan)
SAVE_DIR      = "/workspace/AggNet/outputs/dataset4"
HOST          = "0.0.0.0"
PORT          = 5000
IMG_SIZE      = (224, 224)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHT_MIN = 400.0
WEIGHT_MAX = 900.0
AGG_TYPE_MAP = {'Aggregate 3_4inch': 0.0, 'Aggregate 3_8inch': 1.0}

# ── Sieve config ──
SIEVE_COLS   = ['3_4inch', '1_2inch', '3_8inch', 'No4', 'No8', 'Pan']
SIEVE_LABELS = ['3/4"', '1/2"', '3/8"', '#4', '#8', 'Pan']
SIEVE_MM     = [19.00, 12.50, 9.50, 4.75, 2.36, 0.00]
NUM_SIEVES   = 6

# ── ASTM C-33 Boundaries ──
ASTM_X_LABELS = ['1"',   '3/4"', '1/2"', '3/8"', 'No.4', 'No.8', 'Pan']
ASTM_COARSE   = [100,    90,     55,     20,     0,      0,      0  ]
ASTM_FINE     = [100,    100,    75,     55,     10,     5,      0  ]
CTRL_COARSE   = [100,    90,     57,     28,     3,      2,      0  ]
CTRL_FINE     = [100,    100,    71,     42,     13,     8,      0  ]

# ── Production Ratio Limits ──
ZONE_LIMITS = {
    '3/4"': {'astm': (25, 45), 'ctrl': (35, 43), 'avg': 39},
    '1/2"': {'astm': (20, 35), 'ctrl': (29, 35), 'avg': 32},
    '3/8"': {'astm': (20, 55), 'ctrl': (25, 33), 'avg': 29},
}

os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DATABASE (Supabase)
# ─────────────────────────────────────────────
try:
    import db as aggnet_db
    aggnet_db.init_db()
    DB_ENABLED = True
    print("DB: Supabase connected")
except Exception as _db_err:
    DB_ENABLED = False
    print(f"DB: disabled ({_db_err})")

# ─────────────────────────────────────────────
# UPLOAD FOLDER (เก็บรูปภาพหินตาม doc_no)
# ─────────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_image_file(image_file, company_id, doc_no):
    """บันทึกรูปภาพลงดิสก์ คืน path สัมพัทธ์"""
    folder = os.path.join(UPLOAD_DIR, str(company_id))
    os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(image_file.filename)[1] or '.jpg'
    fname = f"{doc_no}{ext}"
    fpath = os.path.join(folder, fname)
    image_file.stream.seek(0)
    image_file.save(fpath)
    return fpath

def get_image_path(company_id, doc_no, original_filename=''):
    """หา path รูปภาพจาก doc_no"""
    folder = os.path.join(UPLOAD_DIR, str(company_id))
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        p = os.path.join(folder, f"{doc_no}{ext}")
        if os.path.exists(p):
            return p
    return None

# ─────────────────────────────────────────────
# MODEL (import from aggnet_dataset4.py)
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggnet_dataset4 import EfficientNetAggNet

def load_model(path, num_sieves, multitask=False):
    model = EfficientNetAggNet(num_sieves=num_sieves, multitask=multitask).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    model.eval()
    return model

MODEL_34 = load_model(MODEL_PATH_34, num_sieves=6, multitask=True)
MODEL_38 = load_model(MODEL_PATH_38, num_sieves=3, multitask=False)
print(f"Model 3/4\" loaded: {MODEL_PATH_34}")
print(f"Model 3/8\" loaded: {MODEL_PATH_38}")

TF = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────
def predict_from_image(img_pil, agg_type='Aggregate 3_4inch', weight_g=650.0):
    x = TF(img_pil).unsqueeze(0).to(DEVICE)

    weight_norm = float(np.clip((weight_g - WEIGHT_MIN) / (WEIGHT_MAX - WEIGHT_MIN), 0, 1))
    agg_code    = AGG_TYPE_MAP.get(agg_type, 0.0)
    w = torch.tensor([[weight_norm, agg_code]], dtype=torch.float32).to(DEVICE)

    if agg_type == 'Aggregate 3_8inch':
        # 3_8inch: predict No4/No8/Pan เท่านั้น, hardcode 3 sieve แรก
        with torch.no_grad():
            pred_3 = MODEL_38(x, w).squeeze().cpu().numpy() * 100  # (3,) = No4, No8, Pan
        pred = np.array([100.0, 100.0, 99.7, pred_3[0], pred_3[1], pred_3[2]])
    else:
        with torch.no_grad():
            pred = MODEL_34(x, w).squeeze().cpu().numpy() * 100    # (6,)

    return pred   # shape (6,): 3/4", 1/2", 3/8", No4, No8, Pan

def compute_sieve_table(pred_6):
    """คำนวณตาราง Sieve Analysis ครบ"""
    passing    = np.insert(pred_6, 0, 100.0)   # เพิ่ม 1"=100
    ind_retain = np.zeros(7)
    cu_retain  = np.zeros(7)

    for i in range(7):
        if i == 0:
            ind_retain[i] = 100.0 - passing[i]
        else:
            ind_retain[i] = passing[i-1] - passing[i]
        ind_retain[i] = max(0.0, ind_retain[i])

    cu_retain = np.cumsum(ind_retain)

    return passing, ind_retain, cu_retain

def compute_production_ratio(ind_retain):
    """คำนวณ % Retain สะสม 3 ช่วง"""
    total = sum(ind_retain)
    if total <= 0:
        return {'3/4"': 0, '1/2"': 0, '3/8"': 0}

    # Normalize
    r = [x / total * 100 for x in ind_retain]

    zone_3_4 = r[0] + r[1] + r[2]   # 1" + 3/4" + 1/2"
    zone_1_2 = r[3]                   # 3/8"
    zone_3_8 = r[4] + r[5] + r[6]   # #4 + #8 + Pan

    return {'3/4"': zone_3_4, '1/2"': zone_1_2, '3/8"': zone_3_8}

def compute_fm(cu_ret_full, agg_type='Aggregate 3_4inch'):
    """คำนวณ Fineness Modulus (FM) จาก % Cumulative Retain
    cu_ret_full: array 7 ค่า [1", 3/4", 1/2", 3/8", #4, #8, Pan]

    หิน 3/4" (coarse):
      FM = (3/4" + 1/2" + 3/8" + #4 + #8) / 100  — ไม่รวม Pan
    หิน 3/8" (fine):
      FM = (3/8" + #4 + #8 + Pan) / 100  — Pan แทน #16
    """
    if '3_8' in agg_type:
        # index: 3/8"=3, #4=4, #8=5, Pan=6
        return sum([cu_ret_full[3], cu_ret_full[4], cu_ret_full[5], cu_ret_full[6]]) / 100.0
    else:
        # index: 3/4"=1, 1/2"=2, 3/8"=3, #4=4, #8=5  (ไม่รวม Pan=6)
        return sum(cu_ret_full[1:6]) / 100.0

def get_zone_status(val, zone_name):
    lim = ZONE_LIMITS[zone_name]
    if lim['ctrl'][0] <= val <= lim['ctrl'][1]:
        return 'In Control', colors.green
    elif lim['astm'][0] <= val <= lim['astm'][1]:
        return 'In ASTM', colors.orange
    else:
        return 'Out of Range', colors.red


# ─────────────────────────────────────────────
# CHART GENERATION
# ─────────────────────────────────────────────
def generate_chart(pred_6, source_name="", agg_type='Aggregate 3_4inch'):
    """สร้าง Gradation Curve และ return เป็น bytes"""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')

    if agg_type == 'Aggregate 3_8inch':
        # 3/8" aggregate: แสดงเฉพาะ 1/2", 3/8", #4, #8, Pan (5 sieves)
        # pred_6 = [3/4"=0, 1/2"=100, 3/8"≈99.7, No4, No8, Pan]
        passing = pred_6[1:]  # ตัด 3/4" ออก → [1/2", 3/8", #4, #8, Pan]
        x_labels = ['1/2"', '3/8"', 'No.4', 'No.8', 'Pan']
        x = list(range(len(x_labels)))
        # ASTM C33 #8 stone (3/8" nominal) limits
        astm_c = [100, 85, 10, 0, 0]
        astm_f = [100, 100, 30, 10, 5]
        ctrl_c = [100, 85, 10, 0, 0]
        ctrl_f = [100, 100, 30, 10, 5]

        ax.plot(x, astm_c, 'k-^', linewidth=1.5, markersize=5, label='Coarse-ASTM', alpha=0.8)
        ax.plot(x, astm_f, 'k-v', linewidth=1.5, markersize=5, label='Fine-ASTM', alpha=0.8)
        ax.fill_between(x, astm_c, astm_f, alpha=0.07, color='gray')
        ax.plot(x, ctrl_c, 'b--', linewidth=1.0, alpha=0.6, label='Coarse-Control')
        ax.plot(x, ctrl_f, 'b:',  linewidth=1.0, alpha=0.6, label='Fine-Control')
        ax.plot(x, passing, 'r-o', linewidth=2.5, markersize=7, label='Result', zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_title('Gradation Of Fine Aggregate (3/8")', fontsize=13, fontweight='bold')
    else:
        # 3/4" aggregate: แสดงครบ 7 sieves (1" ถึง Pan)
        passing = np.insert(pred_6, 0, 100.0)
        x = list(range(7))
        ax.plot(x, ASTM_COARSE, 'k-^', linewidth=1.5, markersize=5, label='Coarse-ASTM', alpha=0.8)
        ax.plot(x, ASTM_FINE,   'k-v', linewidth=1.5, markersize=5, label='Fine-ASTM', alpha=0.8)
        ax.fill_between(x, ASTM_COARSE, ASTM_FINE, alpha=0.07, color='gray')
        ax.plot(x, CTRL_COARSE, 'b--', linewidth=1.0, alpha=0.6, label='Coarse-Control')
        ax.plot(x, CTRL_FINE,   'b:',  linewidth=1.0, alpha=0.6, label='Fine-Control')
        ax.plot(x, passing, 'r-o', linewidth=2.5, markersize=7, label='Result', zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(ASTM_X_LABELS, fontsize=10)
        ax.set_title('Gradation Of Coarse Aggregate', fontsize=13, fontweight='bold')

    ax.set_yticks(range(0, 110, 10))
    ax.set_ylim(-5, 110)
    ax.set_xlabel('Sieve Size', fontsize=11)
    ax.set_ylabel('% Passing', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return buf


def generate_compare_chart(ai_pass, lab_pass, agg_type='Aggregate 3_4inch'):
    """สร้าง Dual-Line Gradation Chart เปรียบเทียบ AI vs Lab"""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')

    if agg_type == 'Aggregate 3_8inch':
        # 3/8": แสดงเฉพาะ 1/2" ถึง Pan (5 sieve)
        x_labels = ['1/2"', '3/8"', 'No.4', 'No.8', 'Pan']
        x = list(range(len(x_labels)))
        astm_c = [100, 85, 10, 0, 0]
        astm_f = [100, 100, 30, 10, 5]
        ai_y  = ai_pass[1:]   # ตัด 1" ออก
        lab_y = lab_pass[1:]
    else:
        x_labels = ASTM_X_LABELS  # ['1"','3/4"','1/2"','3/8"','No.4','No.8','Pan']
        x = list(range(7))
        astm_c = ASTM_COARSE
        astm_f = ASTM_FINE
        ai_y  = ai_pass
        lab_y = lab_pass

    ax.fill_between(x, astm_c, astm_f, alpha=0.10, color='gray', label='ASTM Range')
    ax.plot(x, astm_c, 'k-^', linewidth=1.0, markersize=4, alpha=0.6)
    ax.plot(x, astm_f, 'k-v', linewidth=1.0, markersize=4, alpha=0.6)
    ax.plot(x, ai_y,  'b-o',  linewidth=2.5, markersize=7, label='AI Predict', zorder=5)
    ax.plot(x, lab_y, 'g-s',  linewidth=2.5, markersize=7, label='Lab จริง',  zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticks(range(0, 110, 10))
    ax.set_ylim(-5, 110)
    ax.set_xlabel('Sieve Size', fontsize=11)
    ax.set_ylabel('% Passing', fontsize=11)
    title = 'Gradation Curve — AI vs Lab' + (' (3/8")' if '3_8' in agg_type else ' (3/4")')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return buf


def generate_compare_pdf(row, sieve_json, label_json):
    """สร้าง PDF เปรียบเทียบ AI vs Lab พร้อมตาราง, chart dual-line และ MAE summary"""
    import json as _json
    from reportlab.lib import colors as rl_colors

    # ── parse ──────────────────────────────────────────────────────────────────
    if isinstance(sieve_json, str): sieve_json = _json.loads(sieve_json)
    if isinstance(label_json, str): label_json = _json.loads(label_json)

    sieve_names = ['1"', '3/4"', '1/2"', '3/8"', '#4', '#8', 'Pan']
    SIEVE_LABEL_MAP = {'1"':0,'3/4"':1,'1/2"':2,'3/8"':3,'#4':4,'#8':5,'Pan':6}

    # AI %pass จาก sieve_json (7 ค่า: 1"=100 + 6 ค่าที่ predict)
    ai_pass = [s['passing'] for s in sieve_json]   # 7 ค่า
    # Lab %pass จาก label_json
    lab_map = {s['sieve']: s['passing'] for s in label_json.get('sieves', [])
               if s.get('passing') is not None}
    lab_pass = [lab_map.get(name) for name in sieve_names]  # None ถ้าไม่มี

    # ── Diff & MAE ──────────────────────────────────────────────────────────────
    diffs = []
    for ai, lab in zip(ai_pass, lab_pass):
        if lab is not None and ai is not None:
            diffs.append(abs(float(ai) - float(lab)))
        else:
            diffs.append(None)
    valid_diffs = [d for d in diffs if d is not None]
    mae = sum(valid_diffs) / len(valid_diffs) if valid_diffs else 0.0
    pass_5 = sum(1 for d in valid_diffs if d <= 5.0)
    total_sieve = len(valid_diffs)

    # ── Chart ──────────────────────────────────────────────────────────────────
    lab_full = [float(lab_map.get(n, 0)) if lab_map.get(n) is not None else 0.0
                for n in sieve_names]
    chart_buf = generate_compare_chart(ai_pass, lab_full, row.get('agg_type',''))

    # ── PDF setup ──────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=15*mm, bottomMargin=15*mm,
                            leftMargin=20*mm, rightMargin=20*mm)

    s_title  = ParagraphStyle('t',  fontSize=14, fontName=_THAI_FONT, alignment=TA_CENTER, spaceAfter=4)
    s_hdr    = ParagraphStyle('h',  fontSize=9,  fontName=_THAI_FONT)
    s_body   = ParagraphStyle('b',  fontSize=8.5,fontName=_THAI_FONT)
    s_small  = ParagraphStyle('sm', fontSize=8,  fontName=_THAI_FONT, textColor=rl_colors.grey)
    s_note   = ParagraphStyle('n',  fontSize=8,  fontName=_THAI_FONT, leftIndent=8, spaceAfter=2)
    s_warn   = ParagraphStyle('w',  fontSize=8,  fontName=_THAI_FONT, textColor=rl_colors.red, leftIndent=8)

    doc_no      = row.get('doc_no') or ''
    tested_date = str(row.get('tested_date') or '')
    tested_time = row.get('tested_time') or ''
    source      = row.get('source') or ''
    agg_type    = row.get('agg_type') or ''
    agg_label   = 'หิน 3/4"' if '3_4' in agg_type else 'หิน 3/8"'
    weight_g    = label_json.get('weight_total_g')

    story = []

    # ── Header ─────────────────────────────────────────────────────────────────
    story.append(Paragraph('<b>รายงานเปรียบเทียบผลการวิเคราะห์</b>', s_title))
    story.append(Paragraph('<b>AI Prediction vs Lab Sieve Analysis</b>', s_title))
    story.append(Spacer(1, 2*mm))
    story.append(HRFlowable(width='100%', thickness=1, color=rl_colors.HexColor('#2b6cb0')))
    story.append(Spacer(1, 2*mm))

    meta = [
        [Paragraph(f'<b>เลขที่เอกสาร:</b> {doc_no}', s_hdr),
         Paragraph(f'<b>วันที่ทดสอบ:</b> {tested_date}  {tested_time}', s_hdr)],
        [Paragraph(f'<b>แหล่งหิน:</b> {source}', s_hdr),
         Paragraph(f'<b>ประเภทหิน:</b> {agg_label}', s_hdr)],
    ]
    if weight_g:
        meta.append([Paragraph(f'<b>น้ำหนักตัวอย่าง:</b> {weight_g} g', s_hdr), Paragraph('', s_hdr)])
    meta_tbl = Table(meta, colWidths=[85*mm, 85*mm])
    meta_tbl.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'), ('ROWBACKGROUNDS',(0,0),(-1,-1),[rl_colors.white, rl_colors.Color(0.95,0.97,1)])]))
    story.append(meta_tbl)
    story.append(Spacer(1, 3*mm))

    # ── Comparison Table ────────────────────────────────────────────────────────
    story.append(Paragraph('<b>ตารางเปรียบเทียบ %Passing</b>', s_hdr))
    story.append(Spacer(1, 1.5*mm))

    tbl_data = [[
        Paragraph('<b>ตะแกรง</b>', s_body),
        Paragraph('<b>AI Predict (%)</b>', s_body),
        Paragraph('<b>Lab จริง (%)</b>', s_body),
        Paragraph('<b>Diff (%)</b>', s_body),
        Paragraph('<b>สถานะ</b>', s_body),
    ]]
    row_styles = []
    for i, name in enumerate(sieve_names):
        ai  = ai_pass[i]
        lab = lab_pass[i]
        diff = diffs[i]
        if lab is None:
            tbl_data.append([
                Paragraph(name, s_body),
                Paragraph(f'{float(ai):.2f}' if ai is not None else '-', s_body),
                Paragraph('-', s_body),
                Paragraph('-', s_body),
                Paragraph('ไม่มีข้อมูล', s_small),
            ])
        else:
            ok = diff <= 5.0
            status = '✅ ผ่าน' if ok else '❌ เกิน 5%'
            if not ok:
                row_styles.append(('BACKGROUND', (0, i+1), (-1, i+1), rl_colors.Color(1, 0.92, 0.92)))
                row_styles.append(('TEXTCOLOR',  (3, i+1), (4, i+1),  rl_colors.red))
            tbl_data.append([
                Paragraph(f'<b>{name}</b>', s_body),
                Paragraph(f'{float(ai):.2f}', s_body),
                Paragraph(f'{float(lab):.2f}', s_body),
                Paragraph(f'{diff:.2f}', s_body),
                Paragraph(status, s_body),
            ])

    comp_tbl = Table(tbl_data, colWidths=[25*mm, 38*mm, 38*mm, 30*mm, 39*mm])
    base_style = [
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), rl_colors.white),
        ('ALIGN',      (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN',      (0, 0), (0, -1), 'CENTER'),
        ('GRID',       (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ('FONTSIZE',   (0, 0), (-1, -1), 8.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.Color(0.95, 0.97, 1)]),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    comp_tbl.setStyle(TableStyle(base_style + row_styles))
    story.append(comp_tbl)
    story.append(Spacer(1, 3*mm))

    # ── MAE Summary (compact) ──────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5, color=rl_colors.grey))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph('<b>สรุปความแม่นยำของ AI (MAE Summary)</b>', s_hdr))
    story.append(Spacer(1, 1.5*mm))

    mae_label = 'ดีมาก' if mae <= 5.0 else ('พอใช้' if mae <= 10.0 else 'ควรปรับปรุง')
    story.append(Paragraph(
        f'<b>MAE (Mean Absolute Error) = {mae:.2f}%  →  {mae_label}</b>  '
        f'| ตะแกรงที่ผ่านเกณฑ์ (Diff ≤ 5%): {pass_5} / {total_sieve} ตะแกรง', s_hdr))
    story.append(Spacer(1, 3*mm))

    # ── Dual-Line Chart ─────────────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5, color=rl_colors.grey))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph('<b>กราฟ Gradation Curve เปรียบเทียบ</b>', s_hdr))
    story.append(Spacer(1, 1.5*mm))
    story.append(RLImage(chart_buf, width=170*mm, height=85*mm))
    story.append(Spacer(1, 1.5*mm))
    story.append(Paragraph(
        'เส้นสีน้ำเงิน = AI Predict  |  เส้นสีเขียว = Lab จริง  |  พื้นสีเทา = ASTM Range',
        s_small))
    story.append(Spacer(1, 5*mm))

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5, color=rl_colors.grey))
    story.append(Paragraph(
        f'QMS-FM-QA-03 : Rev 00 : 02/09/2021  |  AI Comparison Report  |  Generated by AggNet AI',
        ParagraphStyle('ft', fontSize=7, alignment=TA_RIGHT, textColor=rl_colors.grey)))

    doc.build(story)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
# PDF REPORT GENERATION
# ─────────────────────────────────────────────
def to_12h(time_str):
    """แปลง HH:MM หรือ HH:MM:SS → 12:15 PM"""
    try:
        parts = time_str.split(':')
        h, m = int(parts[0]), int(parts[1])
        period = 'AM' if h < 12 else 'PM'
        h12 = h % 12 or 12
        return f"{h12}:{m:02d} {period}"
    except Exception:
        return time_str

def generate_pdf(pred_6, source, agg_type, tested_date, tested_time, chart_buf, doc_no=None, image_path=None):
    """สร้าง PDF Report ตาม format SIEVE ANALYSIS AGGREGATE"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=12*mm, bottomMargin=12*mm)
    styles  = getSampleStyleSheet()
    story   = []
    is_38 = (agg_type == 'Aggregate 3_8inch')

    # ── Fonts & Styles ──
    style_title  = ParagraphStyle('title',  fontSize=13, fontName='Helvetica-Bold',
                                  alignment=TA_CENTER, spaceAfter=2)
    style_sub    = ParagraphStyle('sub',    fontSize=11, fontName='Helvetica-Bold',
                                  alignment=TA_CENTER, spaceAfter=6)
    style_header = ParagraphStyle('hdr',    fontSize=9,  fontName=_THAI_FONT)
    style_small  = ParagraphStyle('sm',     fontSize=8,  fontName=_THAI_FONT)

    # ── Header ──
    doc_no_str = f'เลขที่: {doc_no}' if doc_no else ''
    header_data = [
        [Paragraph('Value Based<br/>Technical Service', style_small),
         Paragraph('<b>SIEVE ANALYSIS RESULT</b>', style_title),
         Paragraph(f'<b>{doc_no_str}</b>', style_small)],
    ]
    header_tbl = Table(header_data, colWidths=[45*mm, 100*mm, 45*mm])
    header_tbl.setStyle(TableStyle([
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(header_tbl)
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    story.append(Spacer(1, 4*mm))

    # ── Info Row ──
    info_data = [
        [Paragraph(f'Source : {source}', style_header),
         Paragraph(f'Tested Date : {tested_date}', style_header)],
        [Paragraph(f'Aggregate Type : {agg_type}', style_header),
         Paragraph(f'Tested Time : {tested_time}', style_header)],
    ]
    info_tbl = Table(info_data, colWidths=[120*mm, 65*mm])
    info_tbl.setStyle(TableStyle([('BOTTOMPADDING', (0,0), (-1,-1), 2)]))
    story.append(info_tbl)
    story.append(Spacer(1, 4*mm))

    # ── Title ──
    story.append(Paragraph('SIEVE  ANALYSIS  AGGREGATE', style_title))
    story.append(Paragraph('ASTM C-33', style_sub))

    def cell(txt, bold=False, align=TA_CENTER, color=None):
        s = ParagraphStyle('c', fontSize=8.5,
                           fontName='Helvetica-Bold' if bold else 'Helvetica',
                           alignment=align, textColor=color or colors.black)
        return Paragraph(str(txt), s)

    # ── Sieve Table ──
    passing_full, ind_ret_full, cu_ret_full = compute_sieve_table(pred_6)

    if is_38:
        # 3/8" aggregate: แสดงเฉพาะ 1/2", 3/8", #4, #8, Pan (ตัด 1" และ 3/4" ออก)
        sieve_names = ['1/2"', '3/8 "', '#4', '#8', 'Pan']
        sizes_mm    = [12.50, 9.50, 4.75, 2.36, '']
        idx_range   = [2, 3, 4, 5, 6]  # index ใน passing_full (0=1", 1=3/4", 2=1/2", ...)
        passing     = [passing_full[i] for i in idx_range]
        # Recalculate ind_ret and cu_ret for the subset
        ind_ret = [0.0] * 5
        for j, i in enumerate(idx_range):
            if j == 0:
                ind_ret[j] = 100.0 - passing[j]
            else:
                ind_ret[j] = max(0.0, passing[j-1] - passing[j])
        cu_ret = list(np.cumsum(ind_ret))
        # ASTM C33 #8 stone limits for these 5 sieves
        astm_c = [100, 85, 10, 0, 0]
        astm_f = [100, 100, 30, 10, 5]
        ctrl_c = [100, 85, 10, 0, 0]
        ctrl_f = [100, 100, 30, 10, 5]
        n_rows = 5
    else:
        sieve_names = ['1 "', '3/4 "', '1/2"', '3/8 "', '#4', '#8', 'Pan']
        sizes_mm    = [25.00, 19.00, 12.50, 9.50, 4.75, 2.36, '']
        passing     = list(passing_full)
        ind_ret     = list(ind_ret_full)
        cu_ret      = list(cu_ret_full)
        astm_c      = ASTM_COARSE
        astm_f      = ASTM_FINE
        ctrl_c      = CTRL_COARSE
        ctrl_f      = CTRL_FINE
        n_rows      = 7

    # Header row
    tbl_data = [[
        cell('Sieve', bold=True), cell('Size\nmm.', bold=True),
        cell('Passing\n%', bold=True), cell('Ind. Ret\n%', bold=True),
        cell('Cu. m. Ret\n%', bold=True), cell('ASTM\nCoarse', bold=True),
        cell('ASTM\nFine', bold=True), cell('Control\nCoarse', bold=True),
        cell('Control\nFine', bold=True),
    ]]

    for i in range(n_rows):
        p_val   = passing[i]
        in_astm = astm_c[i] <= p_val <= astm_f[i]
        p_color = colors.black if in_astm else colors.red
        tbl_data.append([
            cell(sieve_names[i]),
            cell(f'{sizes_mm[i]:.2f}' if sizes_mm[i] != '' else ''),
            cell(f'{p_val:.2f}', color=p_color),
            cell(f'{ind_ret[i]:.2f}'),
            cell(f'{cu_ret[i]:.2f}'),
            cell(str(astm_c[i])),
            cell(str(astm_f[i])),
            cell(str(ctrl_c[i])),
            cell(str(ctrl_f[i])),
        ])

    col_w = [18*mm, 16*mm, 20*mm, 18*mm, 22*mm, 16*mm, 16*mm, 16*mm, 18*mm]
    sieve_tbl = Table(tbl_data, colWidths=col_w)
    sieve_tbl.setStyle(TableStyle([
        ('GRID',        (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND',  (0,0), (-1,0),  colors.lightgrey),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE',    (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.Color(0.95,0.97,1)]),
    ]))
    story.append(sieve_tbl)
    story.append(Spacer(1, 3*mm))

    # ── FM ──
    fm_val = compute_fm(cu_ret_full, agg_type=agg_type)
    fm_data = [[
        cell('Fineness Modulus (FM)', bold=True, align=TA_LEFT),
        cell(f'{fm_val:.2f}', bold=True),
    ]]
    fm_tbl = Table(fm_data, colWidths=[60*mm, 25*mm])
    fm_tbl.setStyle(TableStyle([
        ('GRID',       (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (0,0),   colors.lightblue),
        ('BACKGROUND', (1,0), (1,0),   colors.Color(0.85,0.95,1)),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(fm_tbl)
    story.append(Spacer(1, 4*mm))

    # ── Gradation Chart ──
    chart_buf.seek(0)
    chart_img = RLImage(chart_buf, width=170*mm, height=90*mm)
    story.append(chart_img)
    story.append(Spacer(1, 3*mm))

    # ── Production Ratio Analysis (3/4" only — ไม่แสดงสำหรับ 3/8") ──
    if not is_38:
        story.append(Paragraph('Production Ratio Analysis  (% Cumulative Retain)',
                                style_sub))
        story.append(Spacer(1, 2*mm))

        zones      = compute_production_ratio(ind_ret_full)
        zone_names = ['3/4"', '1/2"', '3/8"']
        z34_key, z12_key, z38_key = '3/4"', '1/2"', '3/8"'

        z34 = ZONE_LIMITS['3/4"']
        z12 = ZONE_LIMITS['1/2"']
        z38 = ZONE_LIMITS['3/8"']
        ratio_data = [
            [cell('3/4"', bold=True), cell('1/2"', bold=True), cell('3/8"', bold=True)],
            [cell(f"ASTM : {z34['astm'][1]} - {z34['astm'][0]} %"),
             cell(f"ASTM : {z12['astm'][1]} - {z12['astm'][0]} %"),
             cell(f"ASTM : {z38['astm'][1]} - {z38['astm'][0]} %")],
            [cell(f"Control : {z34['ctrl'][0]}-{z34['ctrl'][1]} % (av{z34['avg']})"),
             cell(f"Control : {z12['ctrl'][0]}-{z12['ctrl'][1]} % (av{z12['avg']})"),
             cell(f"Control : {z38['ctrl'][0]}-{z38['ctrl'][1]} % (av{z38['avg']})")],
            [cell(f"{zones[z34_key]:.0f}%", bold=True),
             cell(f"{zones[z12_key]:.0f}%", bold=True),
             cell(f"{zones[z38_key]:.0f}%", bold=True)],
        ]
        status_row = []
        for zn in zone_names:
            status, clr = get_zone_status(zones[zn], zn)
            status_row.append(cell(status, bold=True, color=clr))
        ratio_data.append(status_row)

        ratio_tbl = Table(ratio_data, colWidths=[60*mm, 60*mm, 60*mm])
        ratio_tbl.setStyle(TableStyle([
            ('GRID',        (0,0), (-1,-1), 0.5, colors.black),
            ('BACKGROUND',  (0,0), (-1,0),  colors.lightgrey),
            ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
            ('FONTSIZE',    (0,0), (-1,-1), 8.5),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.Color(0.95,0.97,1)]),
        ]))
        story.append(ratio_tbl)
        story.append(Spacer(1, 4*mm))

    # ── Sample Image ──
    if image_path and os.path.exists(image_path):
        # หิน 3/4" มีตาราง Production Ratio ด้วย → ขึ้นหน้าใหม่
        if '3_4' in agg_type:
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 4*mm))
        story.append(Paragraph('<b>ภาพตัวอย่างหิน (Sample Image)</b>',
                                ParagraphStyle('img_hd', fontSize=9, fontName=_THAI_FONT)))
        story.append(Spacer(1, 2*mm))
        story.append(RLImage(image_path, width=80*mm, height=60*mm, kind='proportional'))
        story.append(Spacer(1, 3*mm))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph('QMS-FM-QA-03 : Rev 00 : 02/09/2021  |  Generated by AggNet AI',
                            ParagraphStyle('ft', fontSize=7, alignment=TA_RIGHT,
                                           textColor=colors.grey)))

    doc.build(story)
    buf.seek(0)
    return buf





# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "aggnet-qc-secret-2026")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>AggNet QC</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #1a202c; }

  .header {
    background: linear-gradient(135deg, #1a365d, #2b6cb0);
    color: white; padding: 16px 20px;
    display: flex; align-items: center; gap: 12px;
  }
  .header h1 { font-size: 1.2rem; font-weight: 700; }
  .header p  { font-size: 0.75rem; opacity: 0.8; margin-top: 2px; }
  .logo { width: 40px; height: 40px; background: white; border-radius: 8px;
          display: flex; align-items: center; justify-content: center;
          font-weight: 900; color: #2b6cb0; font-size: 0.9rem; }
  .header-right { margin-left: auto; text-align: right; }
  .header-right .company-name { font-size: 0.75rem; opacity: 0.85; margin-bottom: 4px; }
  .nav-links a { color: #90cdf4; text-decoration: none; font-size: 0.8rem;
                 margin-left: 12px; font-weight: 600; }
  .nav-links a:hover { color: white; }

  .container { padding: 16px; max-width: 480px; margin: 0 auto; }

  .card {
    background: white; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 16px; margin-bottom: 16px;
  }
  .card h2 { font-size: 0.9rem; font-weight: 700; color: #2d3748;
             margin-bottom: 12px; padding-bottom: 8px;
             border-bottom: 2px solid #e2e8f0; }

  label { display: block; font-size: 0.8rem; color: #4a5568;
          font-weight: 600; margin-bottom: 4px; margin-top: 10px; }
  input[type=text], input[type=date], input[type=time], select {
    width: 100%; padding: 10px 12px; border: 1.5px solid #e2e8f0;
    border-radius: 8px; font-size: 0.9rem; color: #2d3748;
    background: #f7fafc; transition: border-color 0.2s;
  }
  input:focus, select:focus {
    outline: none; border-color: #3182ce; background: white;
  }

  .upload-zone {
    border: 2px dashed #bee3f8; border-radius: 12px;
    background: #ebf8ff; padding: 24px; text-align: center;
    cursor: pointer; transition: all 0.2s; margin-top: 10px;
  }
  .upload-zone:hover { border-color: #3182ce; background: #e6f0fa; }
  .upload-zone .icon { font-size: 2.5rem; margin-bottom: 8px; }
  .upload-zone p { font-size: 0.85rem; color: #2b6cb0; font-weight: 600; }
  .upload-zone span { font-size: 0.75rem; color: #718096; }
  #fileInput { display: none; }

  #preview-box { display: none; margin-top: 12px; text-align: center; }
  #preview-box img { max-width: 100%; max-height: 200px;
                     border-radius: 8px; border: 2px solid #bee3f8; }
  #preview-box p { font-size: 0.75rem; color: #718096; margin-top: 4px; }

  .btn-analyze {
    width: 100%; padding: 14px; background: linear-gradient(135deg, #2b6cb0, #1a365d);
    color: white; border: none; border-radius: 10px; font-size: 1rem;
    font-weight: 700; cursor: pointer; margin-top: 8px;
    transition: opacity 0.2s; letter-spacing: 0.5px;
  }
  .btn-analyze:hover   { opacity: 0.9; }
  .btn-analyze:disabled { opacity: 0.5; cursor: not-allowed; }

  #loading {
    display: none; text-align: center; padding: 20px;
  }
  .spinner {
    width: 40px; height: 40px; border: 4px solid #e2e8f0;
    border-top: 4px solid #3182ce; border-radius: 50%;
    animation: spin 0.8s linear infinite; margin: 0 auto 10px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #result-card { display: none; }

  .result-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .result-table th {
    background: #2b6cb0; color: white; padding: 7px 10px; text-align: center;
  }
  .result-table td { padding: 6px 10px; border-bottom: 1px solid #e2e8f0;
                     text-align: center; }
  .result-table tr:nth-child(even) td { background: #f7fafc; }
  .out-range { color: #e53e3e; font-weight: 700; }
  .in-ctrl   { color: #38a169; font-weight: 700; }
  .in-astm   { color: #dd6b20; font-weight: 700; }

  .zone-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;
               margin-top: 10px; }
  .zone-card { border-radius: 8px; padding: 10px; text-align: center;
               border: 1.5px solid #e2e8f0; }
  .zone-card .zone-name { font-size: 0.85rem; font-weight: 700; color: #2d3748; }
  .zone-card .zone-val  { font-size: 1.4rem; font-weight: 900; margin: 4px 0; }
  .zone-card .zone-limit { font-size: 0.65rem; color: #718096; line-height: 1.4; }
  .zone-card .zone-status { font-size: 0.7rem; font-weight: 700;
                             padding: 2px 6px; border-radius: 4px; margin-top: 4px;
                             display: inline-block; }
  .status-ctrl  { background: #c6f6d5; color: #276749; }
  .status-astm  { background: #feebc8; color: #744210; }
  .status-out   { background: #fed7d7; color: #822727; }

  #chart-img { width: 100%; border-radius: 8px; margin-top: 10px; }

  .btn-pdf {
    width: 100%; padding: 12px; background: linear-gradient(135deg, #e53e3e, #c53030);
    color: white; border: none; border-radius: 10px; font-size: 0.95rem;
    font-weight: 700; cursor: pointer; margin-top: 8px; display: none;
  }
  .btn-pdf:hover { opacity: 0.9; }

  .tag { display: inline-block; padding: 2px 6px; border-radius: 4px;
         font-size: 0.7rem; font-weight: 700; }
</style>
</head>
<body>

<div class="header">
  <div class="logo">SCG</div>
  <div>
    <h1>AggNet QC</h1>
    <p>Sieve Analysis AI — ASTM C-33</p>
  </div>
  {% if company_name %}
  <div class="header-right">
    <div class="company-name">🏢 {{ company_name }}</div>
    <div class="nav-links">
      <a href="/history">📋 ประวัติ</a>
      <a href="/logout">ออกจากระบบ</a>
    </div>
  </div>
  {% else %}
  <div class="header-right">
    <div class="nav-links">
      <a href="/login">🔑 เข้าสู่ระบบ</a>
    </div>
  </div>
  {% endif %}
</div>

<div class="container">

  <!-- Input Card -->
  <div class="card">
    <h2>📋 ข้อมูลตัวอย่าง</h2>
    <label>แหล่งหิน (Source)</label>
    <input type="text" id="source" placeholder="เช่น บริษัท ศิลาสานนท์ จำกัด">
    <label>ประเภทมวลรวม</label>
    <select id="agg_type">
      <option value="Aggregate 3_4inch">Aggregate 3/4" (หิน 3/4 นิ้ว)</option>
      <option value="Aggregate 3_8inch">Aggregate 3/8" (หิน 3/8 นิ้ว)</option>
    </select>
    <label>วันที่ทดสอบ</label>
    <input type="date" id="tested_date">
    <label>เวลาทดสอบ</label>
    <input type="time" id="tested_time">
  </div>

  <!-- Upload Card -->
  <div class="card">
    <h2>📷 อัพโหลดภาพหิน</h2>
    <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
      <div class="icon">🪨</div>
      <p>แตะเพื่อเลือกภาพ หรือถ่ายภาพ</p>
      <span>รองรับ JPG, PNG — ภาพรวมหินก่อนร่อน</span>
    </div>
    <input type="file" id="fileInput" accept="image/*"
           onchange="handleFile(this)">
    <div id="preview-box">
      <img id="preview-img" src="" alt="preview">
      <p id="preview-name"></p>
    </div>
  </div>

  <button class="btn-analyze" id="analyzeBtn" onclick="analyze()" disabled>
    🔬 วิเคราะห์ด้วย AI
  </button>

  <!-- Loading -->
  <div id="loading">
    <div class="spinner"></div>
    <p style="color:#4a5568; font-size:0.9rem;">กำลังวิเคราะห์...</p>
  </div>

  <!-- Result -->
  <div class="card" id="result-card">
    <h2>📊 ผลการวิเคราะห์</h2>

    <table class="result-table" id="sieve-table">
      <thead>
        <tr>
          <th>Sieve</th><th>Passing %</th><th>Ind.Ret %</th><th>Cu.Ret %</th>
        </tr>
      </thead>
      <tbody id="sieve-tbody"></tbody>
    </table>

    <div id="fm-box" style="display:none; margin-top:14px; padding:10px 14px;
         background:#ebf8ff; border-radius:10px; border-left:4px solid #3182ce;">
      <span style="font-size:0.8rem; color:#2b6cb0; font-weight:600;">Fineness Modulus (FM)</span>
      <span id="fm-val" style="font-size:1.4rem; font-weight:800; color:#1a365d; margin-left:12px;"></span>
    </div>

    <div id="prod-section" style="display:none">
    <h2 style="margin-top:14px;">🏗️ Production Ratio Analysis</h2>
    <div class="zone-grid" id="zone-grid"></div>
    </div>

    <h2 style="margin-top:14px;">📈 Gradation Curve</h2>
    <img id="chart-img" src="" alt="chart">
  </div>

  <button class="btn-pdf" id="pdfBtn" onclick="downloadPDF()">
    📄 Download PDF Report
  </button>

</div>

<script>
let currentFile = null;
let resultData  = null;

// Set today's date and current time
document.getElementById('tested_date').valueAsDate = new Date();
(function() {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2,'0');
  const mm = String(now.getMinutes()).padStart(2,'0');
  document.getElementById('tested_time').value = hh + ':' + mm;
})();

function handleFile(input) {
  if (!input.files[0]) return;
  currentFile = input.files[0];
  const url   = URL.createObjectURL(currentFile);
  document.getElementById('preview-img').src = url;
  document.getElementById('preview-name').textContent = currentFile.name;
  document.getElementById('preview-box').style.display = 'block';
  document.getElementById('analyzeBtn').disabled = false;
}

async function analyze() {
  if (!currentFile) return;

  document.getElementById('loading').style.display    = 'block';
  document.getElementById('result-card').style.display = 'none';
  document.getElementById('pdfBtn').style.display      = 'none';
  document.getElementById('analyzeBtn').disabled       = true;

  const form = new FormData();
  form.append('image',       currentFile);
  form.append('source',      document.getElementById('source').value || 'Unknown');
  form.append('agg_type',    document.getElementById('agg_type').value);
  form.append('tested_date', document.getElementById('tested_date').value);
  form.append('tested_time', document.getElementById('tested_time').value);

  try {
    const res  = await fetch('/analyze', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    resultData = data;
    renderResults(data);
  } catch(e) {
    alert('Connection error: ' + e);
  } finally {
    document.getElementById('loading').style.display  = 'none';
    document.getElementById('analyzeBtn').disabled    = false;
  }
}

function renderResults(data) {
  // Sieve table
  const tbody = document.getElementById('sieve-tbody');
  tbody.innerHTML = '';
  data.sieve_table.forEach(row => {
    const inRange = row.passing >= row.astm_coarse && row.passing <= row.astm_fine;
    tbody.innerHTML += `<tr>
      <td><b>${row.sieve}</b></td>
      <td class="${inRange ? '' : 'out-range'}">${row.passing.toFixed(2)}</td>
      <td>${row.ind_ret.toFixed(2)}</td>
      <td>${row.cu_ret.toFixed(2)}</td>
    </tr>`;
  });

  // Zone cards
  const zg = document.getElementById('zone-grid');
  zg.innerHTML = '';
  data.zones.forEach(z => {
    const cls = z.status === 'In Control' ? 'status-ctrl' :
                z.status === 'In ASTM'    ? 'status-astm' : 'status-out';
    zg.innerHTML += `
      <div class="zone-card">
        <div class="zone-name">${z.name}</div>
        <div class="zone-val" style="color:${z.status==='In Control'?'#38a169':z.status==='In ASTM'?'#dd6b20':'#e53e3e'}">${z.value.toFixed(0)}%</div>
        <div class="zone-limit">ASTM: ${z.astm_min}-${z.astm_max}%<br>Ctrl: ${z.ctrl_min}-${z.ctrl_max}% (av${z.avg})</div>
        <span class="zone-status ${cls}">${z.status}</span>
      </div>`;
  });

  // Production Ratio — ซ่อนสำหรับ 3/8"
  const is38 = data.agg_type === 'Aggregate 3_8inch';
  document.getElementById('prod-section').style.display = is38 ? 'none' : 'block';

  // FM
  if (data.fm !== undefined) {
    document.getElementById('fm-val').textContent = data.fm.toFixed(2);
    document.getElementById('fm-box').style.display = 'block';
  }

  // Chart
  document.getElementById('chart-img').src = 'data:image/png;base64,' + data.chart_b64;

  document.getElementById('result-card').style.display = 'block';
  document.getElementById('pdfBtn').style.display      = 'block';
  document.getElementById('result-card').scrollIntoView({ behavior: 'smooth' });
}

async function downloadPDF() {
  if (!currentFile || !resultData) return;
  const form = new FormData();
  form.append('image',       currentFile);
  form.append('source',      document.getElementById('source').value || 'Unknown');
  form.append('agg_type',    document.getElementById('agg_type').value);
  form.append('tested_date', document.getElementById('tested_date').value);
  form.append('tested_time', document.getElementById('tested_time').value);

  const res  = await fetch('/report', { method: 'POST', body: form });
  const blob = await res.blob();
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `AggNet_Report_${new Date().toISOString().slice(0,10)}.pdf`;
  a.click();
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE,
        company_name=session.get('company_name', ''))

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file        = request.files['image']
        source      = request.form.get('source', 'Unknown')
        agg_type    = request.form.get('agg_type', 'Aggregate 3_4inch')
        tested_date = request.form.get('tested_date', str(datetime.date.today()))
        tested_time = request.form.get('tested_time', '08:00')

        img  = Image.open(file.stream).convert('RGB')
        pred = predict_from_image(img, agg_type=agg_type)

        passing, ind_ret, cu_ret = compute_sieve_table(pred)
        zones_raw = compute_production_ratio(ind_ret)

        sieve_names_full = ['1 "', '3/4 "', '1/2"', '3/8 "', '#4', '#8', 'Pan']
        sieve_table = []
        for i in range(7):
            sieve_table.append({
                'sieve':      sieve_names_full[i],
                'passing':    float(passing[i]),
                'ind_ret':    float(ind_ret[i]),
                'cu_ret':     float(cu_ret[i]),
                'astm_coarse': ASTM_COARSE[i],
                'astm_fine':   ASTM_FINE[i],
            })

        zones_out = []
        for zn in ['3/4"', '1/2"', '3/8"']:
            val    = zones_raw[zn]
            status, _ = get_zone_status(val, zn)
            lim    = ZONE_LIMITS[zn]
            zones_out.append({
                'name':     zn,
                'value':    float(val),
                'status':   status,
                'astm_min': lim['astm'][0], 'astm_max': lim['astm'][1],
                'ctrl_min': lim['ctrl'][0], 'ctrl_max': lim['ctrl'][1],
                'avg':      lim['avg'],
            })

        # FM
        fm = compute_fm(cu_ret, agg_type=agg_type)

        # Chart
        chart_buf = generate_chart(pred, source, agg_type=agg_type)
        import base64
        chart_b64 = base64.b64encode(chart_buf.read()).decode()

        # บันทึกลง DB (ถ้า login อยู่)
        _doc_no = None
        if DB_ENABLED and session.get('company_id'):
            try:
                _, _doc_no = aggnet_db.save_analysis(
                    company_id     = session['company_id'],
                    agg_type       = agg_type,
                    source         = source,
                    fm             = round(float(fm), 2),
                    sieve_data     = sieve_table,
                    zones_data     = zones_out,
                    image_filename = file.filename,
                    tested_date    = tested_date,
                    tested_time    = to_12h(tested_time),
                )
                # บันทึกรูปภาพลงดิสก์ชื่อ doc_no
                if _doc_no:
                    save_image_file(file, session['company_id'], _doc_no)
            except Exception as _db_save_err:
                print(f"DB save error: {_db_save_err}")  # ไม่ crash app

        return jsonify({
            'sieve_table': sieve_table,
            'zones':       zones_out,
            'chart_b64':   chart_b64,
            'fm':          round(float(fm), 2),
            'agg_type':    agg_type,
            'doc_no':      _doc_no,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report', methods=['POST'])
def report():
    try:
        file        = request.files['image']
        source      = request.form.get('source', 'Unknown')
        agg_type    = request.form.get('agg_type', 'Aggregate 3_4inch')
        tested_date = request.form.get('tested_date', str(datetime.date.today()))
        tested_time = to_12h(request.form.get('tested_time', '08:00'))

        img  = Image.open(file.stream).convert('RGB')
        pred = predict_from_image(img, agg_type=agg_type)

        chart_buf = generate_chart(pred, source, agg_type=agg_type)
        pdf_buf   = generate_pdf(pred, source, agg_type,
                                  tested_date, tested_time, chart_buf)

        fname = f"AggNet_Report_{tested_date}.pdf"
        pdf_bytes = pdf_buf.read()
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{fname}"'
        response.headers['Content-Length'] = str(len(pdf_bytes))
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────

LOGIN_PAGE = """<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AggNet QC — เข้าสู่ระบบ</title>
  <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Sarabun', sans-serif; background: #f0f4f8;
           display: flex; align-items: center; justify-content: center; min-height: 100vh; }
    .card { background: white; border-radius: 16px; padding: 40px 36px; width: 360px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.10); }
    .logo { text-align: center; font-size: 2rem; margin-bottom: 6px; }
    h1 { text-align: center; font-size: 1.3rem; color: #1a365d; margin-bottom: 24px; }
    label { font-size: 0.85rem; color: #4a5568; font-weight: 600; display: block; margin-bottom: 4px; }
    input { width: 100%; padding: 10px 14px; border: 1.5px solid #e2e8f0;
            border-radius: 8px; font-size: 1rem; font-family: inherit; margin-bottom: 16px; }
    input:focus { outline: none; border-color: #3182ce; }
    .btn { width: 100%; padding: 12px; background: #2b6cb0; color: white; border: none;
           border-radius: 8px; font-size: 1rem; font-family: inherit; font-weight: 700;
           cursor: pointer; margin-top: 4px; }
    .btn:hover { background: #2c5282; }
    .error { background: #fff5f5; border: 1px solid #fc8181; color: #c53030;
             border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; font-size: 0.9rem; }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🔬</div>
    <h1>AggNet QC — เข้าสู่ระบบ</h1>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="POST" action="/login">
      <label>ชื่อผู้ใช้</label>
      <input type="text" name="username" required autocomplete="username">
      <label>รหัสผ่าน</label>
      <input type="password" name="password" required autocomplete="current-password">
      <button class="btn" type="submit">เข้าสู่ระบบ</button>
    </form>
  </div>
</body>
</html>"""

HISTORY_PAGE = """<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AggNet QC — ประวัติการวิเคราะห์</title>
  <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Sarabun', sans-serif; background: #f0f4f8; padding: 20px; }
    .topbar { display: flex; align-items: center; justify-content: space-between;
              background: #1a365d; color: white; padding: 12px 20px; border-radius: 12px;
              margin-bottom: 20px; }
    .topbar h1 { font-size: 1.1rem; }
    .topbar a { color: #90cdf4; text-decoration: none; font-size: 0.9rem; }
    .filter-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; align-items: center; }
    select, input[type=date] { padding: 8px 12px; border: 1.5px solid #e2e8f0;
                                border-radius: 8px; font-family: inherit; font-size: 0.9rem; }
    .btn-sm { padding: 8px 16px; background: #2b6cb0; color: white; border: none;
              border-radius: 8px; font-family: inherit; cursor: pointer; font-size: 0.9rem; }
    .btn-export { background: #276749; }
    table { width: 100%; background: white; border-radius: 12px;
            border-collapse: collapse; overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
    th { background: #2b6cb0; color: white; padding: 10px 14px; text-align: left; font-size: 0.85rem; }
    td { padding: 10px 14px; border-bottom: 1px solid #e2e8f0; font-size: 0.9rem; vertical-align: middle; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #ebf8ff; cursor: pointer; }
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .b34 { background: #bee3f8; color: #2b6cb0; }
    .b38 { background: #c6f6d5; color: #276749; }
    .empty { text-align: center; padding: 40px; color: #718096; }
    .action-btns { display: flex; gap: 6px; flex-wrap: nowrap; }
    .btn-view   { padding: 4px 10px; background: #3182ce; color: white; border: none;
                  border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-family: inherit; }
    .btn-pdf    { padding: 4px 10px; background: #e53e3e; color: white; border: none;
                  border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-family: inherit; }
    .btn-edit   { padding: 4px 10px; background: #dd6b20; color: white; border: none;
                  border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-family: inherit; }
    .btn-del    { padding: 4px 10px; background: #718096; color: white; border: none;
                  border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-family: inherit; }
    /* Modal */
    .modal-overlay { display:none; position:fixed; top:0; left:0; width:100%; height:100%;
                     background:rgba(0,0,0,0.5); z-index:1000; align-items:center; justify-content:center; }
    .modal-overlay.open { display:flex; }
    .modal { background:white; border-radius:16px; padding:28px; width:90%; max-width:680px;
             max-height:90vh; overflow-y:auto; position:relative; }
    .modal h2 { font-size:1.1rem; color:#1a365d; margin-bottom:16px; }
    .modal-close { position:absolute; top:12px; right:16px; background:none; border:none;
                   font-size:1.4rem; cursor:pointer; color:#718096; }
    .detail-meta { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:16px; }
    .detail-meta div { background:#f7fafc; border-radius:8px; padding:8px 12px; font-size:0.9rem; }
    .detail-meta span { color:#718096; font-size:0.8rem; display:block; }
    .fm-box { background:#ebf8ff; border-radius:8px; padding:12px 16px; margin-bottom:16px;
              font-size:1.1rem; font-weight:700; color:#2b6cb0; text-align:center; }
    .dtable { width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:8px; }
    .dtable th { background:#2b6cb0; color:white; padding:7px 10px; text-align:center; }
    .dtable td { padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:center; }
    .dtable tr:hover td { background:#ebf8ff; }
    .out-range { color:#e53e3e; font-weight:700; }
    /* Edit modal */
    .edit-input { width:100%; padding:10px; border:1.5px solid #e2e8f0; border-radius:8px;
                  font-family:inherit; font-size:1rem; margin-bottom:12px; }
    .modal-footer { display:flex; gap:10px; justify-content:flex-end; margin-top:16px; }
    .btn-save { padding:8px 20px; background:#276749; color:white; border:none;
                border-radius:8px; font-family:inherit; cursor:pointer; font-weight:700; }
    .btn-cancel { padding:8px 20px; background:#e2e8f0; color:#4a5568; border:none;
                  border-radius:8px; font-family:inherit; cursor:pointer; }
  </style>
</head>
<body>
  <div class="topbar">
    <h1>🔬 AggNet QC — ประวัติ: {{ company_name }}</h1>
    <div>
      <a href="/">🏠 วิเคราะห์</a> &nbsp;|&nbsp;
      <a href="/logout">ออกจากระบบ</a>
    </div>
  </div>

  <form method="GET" action="/history">
    <div class="filter-row">
      <select name="agg_type">
        <option value="">ทุกประเภท</option>
        <option value="Aggregate 3_4inch" {% if filter_agg=='Aggregate 3_4inch' %}selected{% endif %}>หิน 3/4"</option>
        <option value="Aggregate 3_8inch" {% if filter_agg=='Aggregate 3_8inch' %}selected{% endif %}>หิน 3/8"</option>
      </select>
      <input type="date" name="date_from" value="{{ filter_from }}">
      <input type="date" name="date_to"   value="{{ filter_to }}">
      <button class="btn-sm" type="submit">ค้นหา</button>
      <a href="/export?agg_type={{ filter_agg }}&date_from={{ filter_from }}&date_to={{ filter_to }}">
        <button class="btn-sm btn-export" type="button">⬇ Export CSV</button>
      </a>
    </div>
  </form>

  {% if rows %}
  <table id="historyTable">
    <thead>
      <tr>
        <th onclick="sortTable(0)" style="cursor:pointer" title="เรียง"># <span class="sort-icon">⇅</span></th>
        <th onclick="sortTable(1)" style="cursor:pointer" title="เรียง">เลขที่ <span class="sort-icon">⇅</span></th>
        <th onclick="sortTable(2)" style="cursor:pointer" title="เรียง">วันที่-เวลา <span class="sort-icon">⇅</span></th>
        <th onclick="sortTable(3)" style="cursor:pointer" title="เรียง">ประเภทหิน <span class="sort-icon">⇅</span></th>
        <th onclick="sortTable(4)" style="cursor:pointer" title="เรียง">แหล่ง <span class="sort-icon">⇅</span></th>
        <th onclick="sortTable(5)" style="cursor:pointer" title="เรียง">FM <span class="sort-icon">⇅</span></th>
        <th>สถานะ Lab</th>
        <th>การจัดการ</th>
      </tr>
    </thead>
    <tbody>
      {% for r in rows %}
      <tr>
        <td>{{ loop.index }}</td>
        <td><b>{{ r.doc_no or '-' }}</b></td>
        <td>{% if r.tested_date %}{{ r.tested_date.strftime('%d/%m/%Y') }} {{ r.tested_time or '' }}{% else %}{{ r.analyzed_at.strftime('%d/%m/%Y %H:%M') }}{% endif %}</td>
        <td>
          <span class="badge {{ 'b34' if '3_4' in r.agg_type else 'b38' }}">
            {{ '3/4"' if '3_4' in r.agg_type else '3/8"' }}
          </span>
        </td>
        <td class="source-cell" data-id="{{ r.id }}">{{ r.source }}</td>
        <td><b>{{ '%.2f'|format(r.fm) }}</b></td>
        <td>
          {% if r.has_label %}
            <span class="lab-status done">✅ บันทึกผล Lab แล้ว</span>
          {% else %}
            <span class="lab-status todo">⬜ ยังไม่บันทึกผล Lab</span>
          {% endif %}
        </td>
        <td>
          <div class="action-btns" onclick="event.stopPropagation()">
            <button class="btn-view" onclick="viewDetail({{ r.id }})">🔍 ดูผล</button>
            <button class="btn-pdf"  onclick="downloadPdf({{ r.id }})">📄 PDF</button>
            <button class="btn-edit" onclick="openEdit({{ r.id }}, '{{ r.source|replace("'", "\\'") }}', '{{ r.agg_type }}', '{% if r.tested_date %}{{ r.tested_date.strftime("%Y-%m-%d") }}{% endif %}', '{{ r.tested_time or "" }}')">✏️ แก้ไข</button>
            <button class="btn-del"  onclick="deleteRow({{ r.id }}, this)">🗑️ ลบ</button>
            <button class="btn-lab"  onclick="openLabel({{ r.id }}, '{{ r.doc_no or '' }}')">📊 Lab</button>
            {% if r.has_label %}
            <button class="btn-compare" onclick="openComparePdf({{ r.id }})">📋 ผลวิเคราะห์</button>
            {% else %}
            <button class="btn-compare" disabled title="กรอก Lab ก่อน">📋 ผลวิเคราะห์</button>
            {% endif %}
          </div>
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <!-- Pagination bar -->
  <div id="paginationBar" style="display:flex;align-items:center;justify-content:space-between;margin-top:12px;flex-wrap:wrap;gap:8px">
    <div style="display:flex;align-items:center;gap:8px;font-size:.9rem;color:#4a5568">
      แสดง
      <select id="pageSizeSelect" onchange="setPageSize(parseInt(this.value))"
              style="padding:4px 8px;border:1.5px solid #e2e8f0;border-radius:6px;font-family:inherit;font-size:.9rem">
        <option value="10">10</option>
        <option value="25">25</option>
        <option value="50">50</option>
        <option value="100">100</option>
      </select>
      แถวต่อหน้า &nbsp;|&nbsp; <span id="pageInfo">-</span>
    </div>
    <div style="display:flex;gap:4px" id="pageButtons"></div>
  </div>

  {% else %}
  <div class="empty">ยังไม่มีประวัติการวิเคราะห์</div>
  {% endif %}

  <!-- Detail Modal -->
  <div class="modal-overlay" id="detailModal">
    <div class="modal">
      <button class="modal-close" onclick="closeModal('detailModal')">✕</button>
      <h2 id="modalTitle">ผลการวิเคราะห์</h2>
      <div class="detail-meta" id="modalMeta"></div>
      <div class="fm-box" id="modalFM"></div>
      <table class="dtable">
        <thead>
          <tr><th>ตะแกรง</th><th>%ผ่าน</th><th>Ind.Ret%</th><th>Cu.Ret%</th></tr>
        </thead>
        <tbody id="modalSieve"></tbody>
      </table>
    </div>
  </div>

  <!-- Edit / Re-analyze Modal -->
  <div class="modal-overlay" id="editModal">
    <div class="modal" style="max-width:460px">
      <button class="modal-close" onclick="closeModal('editModal')">✕</button>
      <h2>✏️ วิเคราะห์ใหม่</h2>
      <input type="hidden" id="editId">
      <label style="font-size:.85rem;color:#4a5568;font-weight:600;display:block;margin-bottom:4px">แหล่งหิน (Source)</label>
      <input class="edit-input" type="text" id="editSource" style="margin-bottom:14px">
      <label style="font-size:.85rem;color:#4a5568;font-weight:600;display:block;margin-bottom:4px">ประเภทหิน</label>
      <select id="editAggType" style="width:100%;padding:10px;border:1.5px solid #e2e8f0;border-radius:8px;font-family:inherit;font-size:1rem;margin-bottom:14px">
        <option value="Aggregate 3_4inch">หิน 3/4"</option>
        <option value="Aggregate 3_8inch">หิน 3/8"</option>
      </select>
      <div style="display:flex;gap:10px;margin-bottom:14px">
        <div style="flex:1">
          <label style="font-size:.85rem;color:#4a5568;font-weight:600;display:block;margin-bottom:4px">วันที่ทดสอบ</label>
          <input type="date" id="editDate" style="width:100%;padding:10px;border:1.5px solid #e2e8f0;border-radius:8px;font-family:inherit;font-size:1rem">
        </div>
        <div style="flex:1">
          <label style="font-size:.85rem;color:#4a5568;font-weight:600;display:block;margin-bottom:4px">เวลาทดสอบ</label>
          <input type="time" id="editTime" style="width:100%;padding:10px;border:1.5px solid #e2e8f0;border-radius:8px;font-family:inherit;font-size:1rem">
        </div>
      </div>
      <label style="font-size:.85rem;color:#4a5568;font-weight:600;display:block;margin-bottom:4px">อัพโหลดภาพหินใหม่</label>
      <input type="file" id="editImage" accept="image/*"
             style="width:100%;padding:8px;border:1.5px solid #e2e8f0;border-radius:8px;font-family:inherit;margin-bottom:4px">
      <div id="editStatus" style="font-size:.85rem;color:#718096;min-height:20px;margin-bottom:8px"></div>
      <div class="modal-footer">
        <button class="btn-cancel" onclick="closeModal('editModal')">ยกเลิก</button>
        <button class="btn-save" id="editSaveBtn" onclick="saveReanalyze()">🔄 วิเคราะห์ใหม่</button>
      </div>
    </div>
  </div>

  <!-- Lab Label Modal -->
  <div class="modal-overlay" id="labelModal">
    <div class="modal" style="max-width:480px">
      <button class="modal-close" onclick="closeModal('labelModal')">✕</button>
      <h2>📊 บันทึก Lab Label (Sieve จริง)</h2>
      <p style="font-size:.85rem;color:#718096;margin:0 0 12px">เลขที่เอกสาร: <b id="labelDocNo">-</b></p>
      <input type="hidden" id="labelId">
      <table style="width:100%;border-collapse:collapse;margin-bottom:14px">
        <thead>
          <tr style="background:#2b6cb0;color:#fff">
            <th style="padding:7px 10px;text-align:left">ตะแกรง</th>
            <th style="padding:7px 10px;text-align:center">%ผ่าน</th>
          </tr>
        </thead>
        <tbody id="labelSieveTbody">
          <tr><td style="padding:6px 10px">1"</td>
              <td><input type="number" step="0.01" min="0" max="100" id="lp0" class="lab-inp" placeholder="100"></td></tr>
          <tr><td style="padding:6px 10px">3/4"</td>
              <td><input type="number" step="0.01" min="0" max="100" id="lp1" class="lab-inp" placeholder=""></td></tr>
          <tr><td style="padding:6px 10px">1/2"</td>
              <td><input type="number" step="0.01" min="0" max="100" id="lp2" class="lab-inp" placeholder=""></td></tr>
          <tr><td style="padding:6px 10px">3/8"</td>
              <td><input type="number" step="0.01" min="0" max="100" id="lp3" class="lab-inp" placeholder=""></td></tr>
          <tr><td style="padding:6px 10px">#4</td>
              <td><input type="number" step="0.01" min="0" max="100" id="lp4" class="lab-inp" placeholder=""></td></tr>
          <tr><td style="padding:6px 10px">#8</td>
              <td><input type="number" step="0.01" min="0" max="100" id="lp5" class="lab-inp" placeholder=""></td></tr>
          <tr><td style="padding:6px 10px">Pan</td>
              <td><input type="number" step="0.01" min="0" max="100" id="lp6" class="lab-inp" placeholder="0"></td></tr>
        </tbody>
      </table>
      <div style="margin-bottom:12px;display:flex;align-items:center;gap:12px">
        <label style="font-size:.85rem;color:#4a5568;font-weight:600;white-space:nowrap">น้ำหนักรวม (g)</label>
        <input type="number" step="0.01" min="0" id="labelWeightTotal" class="lab-inp" style="max-width:160px" placeholder="เช่น 500.00">
      </div>
      <div id="labelStatus" style="font-size:.85rem;color:#718096;min-height:18px;margin-bottom:8px"></div>
      <div class="modal-footer">
        <button class="btn-cancel" onclick="closeModal('labelModal')">ยกเลิก</button>
        <button class="btn-save" id="labelSaveBtn" onclick="saveLabel()">💾 บันทึก Label</button>
      </div>
    </div>
  </div>

  <style>
    .lab-inp { width:100%;padding:5px 8px;border:1.5px solid #e2e8f0;border-radius:6px;font-family:inherit;font-size:.9rem;text-align:right }
    .btn-lab  { background:#6b46c1;color:#fff;border:none;border-radius:6px;padding:5px 10px;cursor:pointer;font-size:.82rem;white-space:nowrap }
    .btn-lab:hover { background:#553c9a }
    .btn-compare { background:#276749;color:#fff;border:none;border-radius:6px;padding:5px 10px;cursor:pointer;font-size:.82rem;white-space:nowrap }
    .btn-compare:hover:not([disabled]) { background:#1e4d35 }
    .btn-compare[disabled] { background:#a0aec0;cursor:not-allowed }
    .lab-status { font-size:.78rem;font-weight:600;padding:3px 8px;border-radius:12px;white-space:nowrap }
    .lab-status.done { background:#c6f6d5;color:#276749 }
    .lab-status.todo { background:#fff3cd;color:#856404 }
    #historyTable th[onclick]:hover { background:#1a365d; }
    #historyTable th .sort-icon { opacity:0.5; font-size:.8em; }
    #historyTable th.asc .sort-icon::after  { content:'▲'; }
    #historyTable th.desc .sort-icon::after { content:'▼'; }
    #historyTable th.asc  .sort-icon, #historyTable th.desc .sort-icon { opacity:1; }
    .pg-btn { padding:5px 11px;border:1.5px solid #e2e8f0;border-radius:6px;background:#fff;cursor:pointer;font-size:.85rem;color:#2b6cb0;font-family:inherit }
    .pg-btn:hover { background:#ebf8ff }
    .pg-btn.active { background:#2b6cb0;color:#fff;border-color:#2b6cb0;font-weight:700 }
    .pg-btn:disabled { opacity:.4;cursor:default }
  </style>
  <script>
    // ── Pagination ──────────────────────────────────────────────────────────
    let _pageSize = 10, _curPage = 1;
    let _allRows = [];

    function initPagination() {
      const tbody = document.querySelector('#historyTable tbody');
      if (!tbody) return;
      _allRows = Array.from(tbody.querySelectorAll('tr'));
      // restore saved page size
      const saved = localStorage.getItem('aggnet_pageSize');
      if (saved) {
        _pageSize = parseInt(saved);
        document.getElementById('pageSizeSelect').value = _pageSize;
      }
      _curPage = 1;
      renderPage();
    }

    function setPageSize(n) {
      _pageSize = n;
      _curPage = 1;
      localStorage.setItem('aggnet_pageSize', n);
      renderPage();
    }

    function goPage(p) {
      _curPage = p;
      renderPage();
    }

    function renderPage() {
      const rows = _allRows;  // หลัง sort _allRows อัพเดตแล้ว
      const total = rows.length;
      const totalPages = Math.max(1, Math.ceil(total / _pageSize));
      if (_curPage > totalPages) _curPage = totalPages;
      const start = (_curPage - 1) * _pageSize;
      const end   = Math.min(start + _pageSize, total);
      rows.forEach((tr, i) => {
        tr.style.display = (i >= start && i < end) ? '' : 'none';
      });
      // pageInfo
      document.getElementById('pageInfo').textContent =
        total === 0 ? '0 รายการ' : `${start + 1}–${end} จาก ${total} รายการ`;
      // page buttons
      const bar = document.getElementById('pageButtons');
      bar.innerHTML = '';
      const addBtn = (label, page, disabled, active) => {
        const b = document.createElement('button');
        b.className = 'pg-btn' + (active ? ' active' : '');
        b.textContent = label;
        b.disabled = disabled;
        if (!disabled) b.onclick = () => goPage(page);
        bar.appendChild(b);
      };
      addBtn('‹', _curPage - 1, _curPage === 1, false);
      // แสดงปุ่มหน้าไม่เกิน 7 ปุ่ม
      let lo = Math.max(1, _curPage - 3), hi = Math.min(totalPages, lo + 6);
      if (hi - lo < 6) lo = Math.max(1, hi - 6);
      if (lo > 1) { addBtn('1', 1, false, false); if (lo > 2) bar.appendChild(Object.assign(document.createElement('span'), {textContent:'…', style:'padding:5px 4px;color:#718096;font-size:.85rem'})); }
      for (let p = lo; p <= hi; p++) addBtn(p, p, false, p === _curPage);
      if (hi < totalPages) { if (hi < totalPages - 1) bar.appendChild(Object.assign(document.createElement('span'), {textContent:'…', style:'padding:5px 4px;color:#718096;font-size:.85rem'})); addBtn(totalPages, totalPages, false, false); }
      addBtn('›', _curPage + 1, _curPage === totalPages, false);
    }

    // init เมื่อโหลดหน้า
    document.addEventListener('DOMContentLoaded', initPagination);

    let _sortCol = -1, _sortAsc = true;
    function sortTable(col) {
      const tbl = document.getElementById('historyTable');
      const tbody = tbl.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const ths = tbl.querySelectorAll('thead th');
      if (_sortCol === col) { _sortAsc = !_sortAsc; }
      else { _sortCol = col; _sortAsc = true; }
      ths.forEach((th, i) => { th.classList.remove('asc','desc'); });
      ths[col].classList.add(_sortAsc ? 'asc' : 'desc');
      rows.sort((a, b) => {
        const ta = a.cells[col]?.innerText.trim() || '';
        const tb = b.cells[col]?.innerText.trim() || '';
        const na = parseFloat(ta), nb = parseFloat(tb);
        const cmp = (!isNaN(na) && !isNaN(nb)) ? na - nb : ta.localeCompare(tb, 'th');
        return _sortAsc ? cmp : -cmp;
      });
      rows.forEach(r => tbody.appendChild(r));
      // sync _allRows หลัง sort แล้ว re-render pagination
      _allRows = rows;
      _curPage = 1;
      renderPage();
    }

    function closeModal(id) {
      document.getElementById(id).classList.remove('open');
    }

    async function viewDetail(id) {
      const res = await fetch('/history/' + id + '/detail');
      if (!res.ok) { alert('ไม่พบข้อมูล'); return; }
      const d = await res.json();
      document.getElementById('modalTitle').textContent =
        'ผลการวิเคราะห์ — ' + (d.agg_type.includes('3_4') ? 'หิน 3/4"' : 'หิน 3/8"');
      document.getElementById('modalMeta').innerHTML =
        (d.doc_no ? '<div><span>เลขที่เอกสาร</span><b>' + d.doc_no + '</b></div>' : '') +
        '<div><span>วันที่-เวลา</span>' + d.analyzed_at + '</div>' +
        '<div><span>แหล่งหิน</span>' + (d.source || '-') + '</div>';
      document.getElementById('modalFM').textContent = 'Fineness Modulus (FM): ' + d.fm.toFixed(2);
      const sieves = ['1"','3/4"','1/2"','3/8"','#4','#8','Pan'];
      let rows = '';
      (d.sieve_json || []).forEach((s, i) => {
        rows += '<tr><td>' + sieves[i] + '</td><td>' + s.passing.toFixed(2) +
                '</td><td>' + s.ind_ret.toFixed(2) + '</td><td>' + s.cu_ret.toFixed(2) + '</td></tr>';
      });
      document.getElementById('modalSieve').innerHTML = rows;
      document.getElementById('detailModal').classList.add('open');
    }

    function downloadPdf(id) {
      window.open('/history/' + id + '/pdf', '_blank');
    }

    function openEdit(id, source, aggType, testedDate, testedTime) {
      document.getElementById('editId').value = id;
      document.getElementById('editSource').value = source;
      document.getElementById('editAggType').value = aggType || 'Aggregate 3_4inch';
      document.getElementById('editImage').value = '';
      document.getElementById('editStatus').textContent = '';
      // วันที่/เวลา — ถ้าไม่มีให้ใช้วันปัจจุบัน
      const today = new Date();
      document.getElementById('editDate').value = testedDate ||
        today.toISOString().slice(0,10);
      // แปลง "1:10 PM" → "13:10" สำหรับ input[type=time]
      if (testedTime) {
        const m = testedTime.match(/(\d+):(\d+)\s*(AM|PM)/i);
        if (m) {
          let h = parseInt(m[1]);
          const min = m[2];
          if (m[3].toUpperCase() === 'PM' && h !== 12) h += 12;
          if (m[3].toUpperCase() === 'AM' && h === 12) h = 0;
          document.getElementById('editTime').value = String(h).padStart(2,'0') + ':' + min;
        } else { document.getElementById('editTime').value = testedTime.slice(0,5); }
      } else {
        document.getElementById('editTime').value =
          String(today.getHours()).padStart(2,'0') + ':' + String(today.getMinutes()).padStart(2,'0');
      }
      document.getElementById('editModal').classList.add('open');
    }

    async function saveReanalyze() {
      const id      = document.getElementById('editId').value;
      const src     = document.getElementById('editSource').value.trim();
      const aggType = document.getElementById('editAggType').value;
      const imgFile = document.getElementById('editImage').files[0];
      if (!imgFile) { alert('กรุณาเลือกภาพหินก่อน'); return; }
      const btn = document.getElementById('editSaveBtn');
      btn.disabled = true; btn.textContent = '⏳ กำลังวิเคราะห์...';
      document.getElementById('editStatus').textContent = 'กำลังส่งภาพและวิเคราะห์...';
      const fd = new FormData();
      fd.append('image', imgFile);
      fd.append('source', src);
      fd.append('agg_type', aggType);
      fd.append('tested_date', document.getElementById('editDate').value);
      fd.append('tested_time', document.getElementById('editTime').value);
      const res = await fetch('/history/' + id + '/reanalyze', { method: 'POST', body: fd });
      btn.disabled = false; btn.textContent = '🔄 วิเคราะห์ใหม่';
      if (res.ok) {
        document.getElementById('editStatus').textContent = '✅ อัพเดตสำเร็จ';
        setTimeout(() => { closeModal('editModal'); location.reload(); }, 800);
      } else {
        document.getElementById('editStatus').textContent = '❌ เกิดข้อผิดพลาด';
      }
    }

    async function deleteRow(id, btn) {
      if (!confirm('ยืนยันลบรายการนี้?')) return;
      const res = await fetch('/history/' + id + '/delete', {method: 'POST'});
      if (res.ok) {
        btn.closest('tr').remove();
      } else { alert('ลบไม่สำเร็จ'); }
    }

    function openComparePdf(id) {
      window.open('/history/' + id + '/compare_pdf', '_blank');
    }

    async function openLabel(id, docNo) {
      document.getElementById('labelId').value = id;
      document.getElementById('labelDocNo').textContent = docNo || '-';
      document.getElementById('labelStatus').textContent = '';
      document.getElementById('labelSaveBtn').disabled = false;
      document.getElementById('labelSaveBtn').textContent = '💾 บันทึก Label';
      // ล้างก่อน แล้วค่อย fetch
      for (let i = 0; i <= 6; i++) {
        const pi = document.getElementById('lp' + i);
        if (pi) pi.value = '';
      }
      document.getElementById('labelWeightTotal').value = '';
      document.getElementById('labelModal').classList.add('open');
      // ดึงค่าที่บันทึกไว้ล่าสุด (ถ้ามี)
      try {
        const res = await fetch('/history/' + id + '/label');
        if (res.ok) {
          const d = await res.json();
          if (d && d.sieves && d.sieves.length) {
            d.sieves.forEach((s, i) => {
              const pi = document.getElementById('lp' + i);
              if (pi && s.passing != null) pi.value = s.passing;
            });
          }
          if (d && d.weight_total_g != null) {
            document.getElementById('labelWeightTotal').value = d.weight_total_g;
          }
        }
      } catch(e) { /* ไม่มีค่าเดิม — ปล่อยว่างไว้ */ }
    }

    async function saveLabel() {
      const id = document.getElementById('labelId').value;
      const sieves = ['1"','3/4"','1/2"','3/8"','#4','#8','Pan'];
      const wt = parseFloat(document.getElementById('labelWeightTotal').value);
      const data = { sieves: [], weight_total_g: isNaN(wt) ? null : wt };
      for (let i = 0; i <= 6; i++) {
        const p = parseFloat(parseFloat(document.getElementById('lp' + i).value).toFixed(2));
        data.sieves.push({
          sieve: sieves[i],
          passing: isNaN(p) ? null : p
        });
      }
      const btn = document.getElementById('labelSaveBtn');
      btn.disabled = true; btn.textContent = '⏳ กำลังบันทึก...';
      const res = await fetch('/history/' + id + '/label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
      });
      btn.disabled = false; btn.textContent = '💾 บันทึก Label';
      if (res.ok) {
        document.getElementById('labelStatus').textContent = '✅ บันทึก Label สำเร็จ';
        // อัพเดต badge สถานะในตารางทันที (ไม่ต้อง reload)
        const rows = document.querySelectorAll('#historyTable tbody tr');
        rows.forEach(tr => {
          if (tr.querySelector('.btn-lab[onclick*="openLabel(' + id + ',"]') ||
              tr.innerHTML.includes("openLabel(" + id + ",")) {
            const badge = tr.querySelector('.lab-status');
            if (badge) {
              badge.className = 'lab-status done';
              badge.textContent = '✅ บันทึกผล Lab แล้ว';
            }
          }
        });
        setTimeout(() => closeModal('labelModal'), 1200);
      } else {
        document.getElementById('labelStatus').textContent = '❌ บันทึกไม่สำเร็จ';
      }
    }
  </script>
</body>
</html>"""

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if DB_ENABLED:
            company = aggnet_db.verify_login(username, password)
            if company:
                session['company_id']   = company['id']
                session['company_name'] = company['name']
                return redirect(url_for('index'))
        return render_template_string(LOGIN_PAGE, error='ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง')
    return render_template_string(LOGIN_PAGE, error=None)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/history')
def history():
    if not session.get('company_id'):
        return redirect(url_for('login'))
    filter_agg  = request.args.get('agg_type', '')
    filter_from = request.args.get('date_from', '')
    filter_to   = request.args.get('date_to', '')
    rows = []
    if DB_ENABLED:
        raw = aggnet_db.get_analyses(
            company_id = session['company_id'],
            agg_type   = filter_agg or None,
            date_from  = filter_from or None,
            date_to    = filter_to   or None,
        )
        # แปลง sieve_json string → list
        import json as _json
        for r in raw:
            if isinstance(r['sieve_json'], str):
                r['sieve_json'] = _json.loads(r['sieve_json'])
        rows = raw
    return render_template_string(HISTORY_PAGE,
        company_name = session.get('company_name', ''),
        rows         = rows,
        filter_agg   = filter_agg,
        filter_from  = filter_from,
        filter_to    = filter_to,
    )

@app.route('/history/<int:aid>/detail')
def history_detail(aid):
    if not session.get('company_id'):
        return jsonify({'error': 'unauthorized'}), 401
    if not DB_ENABLED:
        return jsonify({'error': 'db disabled'}), 503
    import json as _json
    row = aggnet_db.get_analysis_by_id(aid, session['company_id'])
    if not row:
        return jsonify({'error': 'not found'}), 404
    sieve = row['sieve_json']
    if isinstance(sieve, str):
        sieve = _json.loads(sieve)
    if row.get('tested_date'):
        display_dt = str(row['tested_date']) + ' ' + (row.get('tested_time') or '')
    else:
        display_dt = row['analyzed_at'].strftime('%d/%m/%Y %H:%M')
    return jsonify({
        'id':          row['id'],
        'doc_no':      row.get('doc_no') or '',
        'analyzed_at': display_dt,
        'agg_type':    row['agg_type'],
        'source':      row['source'],
        'fm':          row['fm'],
        'sieve_json':  sieve,
    })

@app.route('/history/<int:aid>/pdf')
def history_pdf(aid):
    if not session.get('company_id'):
        return redirect(url_for('login'))
    if not DB_ENABLED:
        return 'DB disabled', 503
    import json as _json
    row = aggnet_db.get_analysis_by_id(aid, session['company_id'])
    if not row:
        return 'Not found', 404
    sieve = row['sieve_json']
    if isinstance(sieve, str):
        sieve = _json.loads(sieve)
    # reconstruct pred_6 from sieve_json (indices 1-6 = 3/4" ถึง Pan)
    pred_6 = np.array([s['passing'] for s in sieve[1:]])
    agg_type    = row['agg_type']
    source      = row['source'] or ''
    tested_date = str(row['tested_date']) if row.get('tested_date') else row['analyzed_at'].strftime('%Y-%m-%d')
    tested_time = row.get('tested_time') or to_12h(row['analyzed_at'].strftime('%H:%M'))
    doc_no     = row.get('doc_no') or ''
    image_path = get_image_path(session['company_id'], doc_no) if doc_no else None
    chart_buf  = generate_chart(pred_6, source, agg_type=agg_type)
    pdf_buf    = generate_pdf(pred_6, source, agg_type, tested_date, tested_time, chart_buf,
                              doc_no=doc_no, image_path=image_path)
    fname = f"AggNet_Report_{tested_date}_{aid}.pdf"
    pdf_bytes = pdf_buf.read()
    response = make_response(pdf_bytes)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename="{fname}"'
    return response

@app.route('/history/<int:aid>/edit', methods=['POST'])
def history_edit(aid):
    if not session.get('company_id'):
        return jsonify({'error': 'unauthorized'}), 401
    if not DB_ENABLED:
        return jsonify({'error': 'db disabled'}), 503
    data   = request.get_json()
    source = (data.get('source') or '').strip()
    ok = aggnet_db.update_analysis_source(aid, session['company_id'], source)
    return ('', 200) if ok else ('Not found', 404)

@app.route('/history/<int:aid>/reanalyze', methods=['POST'])
def history_reanalyze(aid):
    if not session.get('company_id'):
        return jsonify({'error': 'unauthorized'}), 401
    if not DB_ENABLED:
        return jsonify({'error': 'db disabled'}), 503
    try:
        file        = request.files['image']
        source      = request.form.get('source', '')
        agg_type    = request.form.get('agg_type', 'Aggregate 3_4inch')
        tested_date = request.form.get('tested_date') or None
        tested_time = to_12h(request.form.get('tested_time', '')) if request.form.get('tested_time') else None
        img  = Image.open(file.stream).convert('RGB')
        pred = predict_from_image(img, agg_type=agg_type)
        passing, ind_ret, cu_ret = compute_sieve_table(pred)
        sieve_labels = ['1"', '3/4"', '1/2"', '3/8"', '#4', '#8', 'Pan']
        sieve_table = []
        for i in range(7):
            sieve_table.append({
                'sieve':   sieve_labels[i],
                'passing': float(passing[i]),
                'ind_ret': float(ind_ret[i]),
                'cu_ret':  float(cu_ret[i]),
            })
        zones_raw = compute_production_ratio(ind_ret)
        zones_out = []
        for zn in ['3/4"', '1/2"', '3/8"']:
            val = zones_raw[zn]
            status, _ = get_zone_status(val, zn)
            lim = ZONE_LIMITS[zn]
            zones_out.append({'name': zn, 'value': float(val), 'status': status,
                               'astm_min': lim['astm'][0], 'astm_max': lim['astm'][1],
                               'ctrl_min': lim['ctrl'][0], 'ctrl_max': lim['ctrl'][1], 'avg': lim['avg']})
        fm = compute_fm(cu_ret, agg_type=agg_type)
        ok = aggnet_db.update_analysis(
            analysis_id=aid, company_id=session['company_id'],
            agg_type=agg_type, source=source,
            fm=round(float(fm), 2), sieve_data=sieve_table, zones_data=zones_out,
            image_filename=file.filename,
            tested_date=tested_date, tested_time=tested_time,
        )
        return ('', 200) if ok else ('Not found', 404)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<int:aid>/delete', methods=['POST'])
def history_delete(aid):
    if not session.get('company_id'):
        return jsonify({'error': 'unauthorized'}), 401
    if not DB_ENABLED:
        return jsonify({'error': 'db disabled'}), 503
    ok = aggnet_db.delete_analysis(aid, session['company_id'])
    return ('', 200) if ok else ('Not found', 404)

@app.route('/history/<int:aid>/label', methods=['GET', 'POST'])
def history_label(aid):
    """GET: ดึง label_json เดิม | POST: บันทึก lab label ใหม่"""
    if not session.get('company_id'):
        return jsonify({'error': 'unauthorized'}), 401
    if not DB_ENABLED:
        return jsonify({'error': 'db disabled'}), 503
    if request.method == 'GET':
        row = aggnet_db.get_analysis_by_id(aid, session['company_id'])
        if not row:
            return jsonify({'error': 'not found'}), 404
        import json as _json
        lj = row.get('label_json')
        if lj is None:
            return jsonify({}), 200
        if isinstance(lj, str):
            lj = _json.loads(lj)
        return jsonify(lj)
    # POST
    import json as _json
    label_data = request.get_json(force=True, silent=True)
    if not label_data:
        return jsonify({'error': 'invalid json'}), 400
    ok = aggnet_db.save_label(aid, session['company_id'], label_data)
    return ('', 200) if ok else ('Not found', 404)

@app.route('/history/<int:aid>/compare_pdf')
def history_compare_pdf(aid):
    """สร้าง PDF เปรียบเทียบ AI vs Lab แล้วส่งกลับ"""
    if not session.get('company_id'):
        return redirect(url_for('login'))
    if not DB_ENABLED:
        return 'DB disabled', 503
    import json as _json
    row = aggnet_db.get_analysis_by_id(aid, session['company_id'])
    if not row:
        return 'Not found', 404
    sj = row.get('sieve_json')
    lj = row.get('label_json')
    if not sj or not lj:
        return 'ไม่มีข้อมูล Lab หรือ AI', 400
    if isinstance(sj, str): sj = _json.loads(sj)
    if isinstance(lj, str): lj = _json.loads(lj)
    pdf_buf = generate_compare_pdf(dict(row), sj, lj)
    doc_no = row.get('doc_no') or str(aid)
    return send_file(
        pdf_buf,
        mimetype='application/pdf',
        as_attachment=False,
        download_name=f'Compare_{doc_no}.pdf'
    )


@app.route('/export')
def export_csv():
    if not session.get('company_id'):
        return redirect(url_for('login'))
    import csv, io as _io, json as _json
    filter_agg  = request.args.get('agg_type', '')
    filter_from = request.args.get('date_from', '')
    filter_to   = request.args.get('date_to', '')
    rows = aggnet_db.get_analyses(
        company_id = session['company_id'],
        agg_type   = filter_agg or None,
        date_from  = filter_from or None,
        date_to    = filter_to   or None,
        limit      = 10000,
    ) if DB_ENABLED else []

    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['DocNo', 'Date', 'Time', 'AggType', 'Source', 'FM',
                     '3/4_%pass', '1/2_%pass', '3/8_%pass', '#4_%pass', '#8_%pass', 'Pan_%pass'])
    for r in rows:
        sieves = r['sieve_json'] if isinstance(r['sieve_json'], list) else _json.loads(r['sieve_json'] or '[]')
        passing = [s['passing'] for s in sieves] if sieves else [''] * 7
        # ใช้ tested_date/tested_time ถ้ามี ไม่งั้น fallback analyzed_at
        if r.get('tested_date'):
            row_date = str(r['tested_date'])
            row_time = r.get('tested_time') or ''
        else:
            row_date = r['analyzed_at'].strftime('%Y-%m-%d')
            row_time = to_12h(r['analyzed_at'].strftime('%H:%M'))
        writer.writerow([
            r.get('doc_no') or '',
            row_date,
            row_time,
            '3/4"' if '3_4' in (r['agg_type'] or '') else '3/8"',
            r['source'],
            f"{r['fm']:.2f}" if r['fm'] else '',
            *[f"{p:.2f}" if p != '' else '' for p in (passing[1:] if len(passing) > 1 else [''] * 6)]
        ])

    out = buf.getvalue().encode('utf-8-sig')  # utf-8-sig สำหรับ Excel ไทย
    resp = make_response(out)
    resp.headers['Content-Type'] = 'text/csv; charset=utf-8'
    resp.headers['Content-Disposition'] = f'attachment; filename="aggnet_history_{datetime.datetime.now().strftime("%Y%m%d")}.csv"'
    return resp

@app.route('/register', methods=['GET', 'POST'])
def register():
    """สร้างบัญชีบริษัทใหม่ (ใช้ภายใน / admin เท่านั้น)"""
    MSG_PAGE = """<!DOCTYPE html><html><head><meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap" rel="stylesheet">
    <style>body{font-family:'Sarabun',sans-serif;display:flex;align-items:center;
    justify-content:center;min-height:100vh;background:#f0f4f8;}
    .card{background:white;border-radius:16px;padding:36px;width:380px;
    box-shadow:0 4px 20px rgba(0,0,0,.1);}
    h1{color:#1a365d;margin-bottom:20px;font-size:1.2rem;}
    label{font-size:.85rem;color:#4a5568;font-weight:600;display:block;margin-bottom:4px;}
    input{width:100%;padding:10px;border:1.5px solid #e2e8f0;border-radius:8px;
    font-family:inherit;font-size:1rem;margin-bottom:14px;}
    .btn{width:100%;padding:12px;background:#276749;color:white;border:none;
    border-radius:8px;font-family:inherit;font-weight:700;cursor:pointer;}
    .msg{padding:10px;border-radius:8px;margin-bottom:14px;font-size:.9rem;}
    .ok{background:#c6f6d5;color:#276749;} .err{background:#fff5f5;color:#c53030;}
    </style></head><body><div class="card">
    <h1>➕ เพิ่มบริษัทใหม่</h1>
    {% if msg %}<div class="msg {{ cls }}">{{ msg }}</div>{% endif %}
    <form method="POST">
      <label>ชื่อบริษัท</label><input name="name" required>
      <label>Username</label><input name="username" required>
      <label>Password</label><input type="password" name="password" required>
      <button class="btn">สร้างบัญชี</button>
    </form></div></body></html>"""

    msg, cls = None, 'ok'
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if DB_ENABLED:
            try:
                aggnet_db.create_company(name, username, password)
                msg, cls = f'สร้างบัญชี "{name}" สำเร็จ', 'ok'
            except Exception as e:
                msg, cls = f'Error: {e}', 'err'
        else:
            msg, cls = 'DB ไม่พร้อม', 'err'
    return render_template_string(MSG_PAGE, msg=msg, cls=cls)

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print(f"  AggNet QC Web App")
    print(f"  http://{HOST}:{PORT}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*50}\n")
    app.run(host=HOST, port=PORT, debug=False)
