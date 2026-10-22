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
                                 Paragraph, Spacer, Image as RLImage, HRFlowable)
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

def compute_fm(cu_ret_full):
    """คำนวณ Fineness Modulus (FM) จาก % Cumulative Retain บนตะแกรง 3/4"–Pan
    FM = (cu_ret[3/4"] + cu_ret[1/2"] + cu_ret[3/8"] + cu_ret[#4] + cu_ret[#8] + cu_ret[Pan]) / 100
    Pan ใช้แทน #16 สำหรับหิน 3/8" (สำหรับหินสะอาด #16 ind.ret ≈ 0 ∴ #16 cu.ret ≈ Pan cu.ret)
    cu_ret_full: array 7 ค่า [1", 3/4", 1/2", 3/8", #4, #8, Pan]
    """
    return sum(cu_ret_full[1:7]) / 100.0  # index 1–6 = 3/4", 1/2", 3/8", #4, #8, Pan

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


# ─────────────────────────────────────────────
# PDF REPORT GENERATION
# ─────────────────────────────────────────────
def generate_pdf(pred_6, source, agg_type, tested_date, tested_time, chart_buf):
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
    header_data = [
        [Paragraph('Value Based<br/>Technical Service', style_small),
         Paragraph('<b>SIEVE ANALYSIS RESULT</b>', style_title),
         Paragraph('', style_small)],
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
    fm_val = compute_fm(cu_ret_full)
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
</div>

<div class="container">

  <!-- Input Card -->
  <div class="card">
    <h2>📋 ข้อมูลตัวอย่าง</h2>
    <label>แหล่งหิน (Source)</label>
    <input type="text" id="source" placeholder="เช่น ART CONCRETE COMPANY LIMITED">
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
    return render_template_string(HTML_PAGE)

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
        fm = compute_fm(cu_ret)

        # Chart
        chart_buf = generate_chart(pred, source, agg_type=agg_type)
        import base64
        chart_b64 = base64.b64encode(chart_buf.read()).decode()

        # บันทึกลง DB (ถ้า login อยู่)
        if DB_ENABLED and session.get('company_id'):
            try:
                aggnet_db.save_analysis(
                    company_id     = session['company_id'],
                    agg_type       = agg_type,
                    source         = source,
                    fm             = round(float(fm), 2),
                    sieve_data     = sieve_table,
                    zones_data     = zones_out,
                    image_filename = image.filename,
                )
            except Exception as _db_save_err:
                print(f"DB save error: {_db_save_err}")  # ไม่ crash app

        return jsonify({
            'sieve_table': sieve_table,
            'zones':       zones_out,
            'chart_b64':   chart_b64,
            'fm':          round(float(fm), 2),
            'agg_type':    agg_type,
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
        tested_time = request.form.get('tested_time', '08:00')

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
    .filter-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
    select, input[type=date] { padding: 8px 12px; border: 1.5px solid #e2e8f0;
                                border-radius: 8px; font-family: inherit; font-size: 0.9rem; }
    .btn-sm { padding: 8px 16px; background: #2b6cb0; color: white; border: none;
              border-radius: 8px; font-family: inherit; cursor: pointer; font-size: 0.9rem; }
    .btn-export { background: #276749; }
    table { width: 100%; background: white; border-radius: 12px;
            border-collapse: collapse; overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
    th { background: #2b6cb0; color: white; padding: 10px 14px; text-align: left; font-size: 0.85rem; }
    td { padding: 10px 14px; border-bottom: 1px solid #e2e8f0; font-size: 0.9rem; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #ebf8ff; }
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .b34 { background: #bee3f8; color: #2b6cb0; }
    .b38 { background: #c6f6d5; color: #276749; }
    .empty { text-align: center; padding: 40px; color: #718096; }
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
  <table>
    <thead>
      <tr>
        <th>#</th><th>วันที่-เวลา</th><th>ประเภทหิน</th><th>แหล่ง</th>
        <th>FM</th><th>#4 (%ผ่าน)</th><th>#8 (%ผ่าน)</th>
      </tr>
    </thead>
    <tbody>
      {% for r in rows %}
      <tr>
        <td>{{ loop.index }}</td>
        <td>{{ r.analyzed_at.strftime('%d/%m/%Y %H:%M') }}</td>
        <td>
          <span class="badge {{ 'b34' if '3_4' in r.agg_type else 'b38' }}">
            {{ '3/4"' if '3_4' in r.agg_type else '3/8"' }}
          </span>
        </td>
        <td>{{ r.source }}</td>
        <td><b>{{ '%.2f'|format(r.fm) }}</b></td>
        <td>{{ '%.1f'|format(r.sieve_json[4].passing) if r.sieve_json else '-' }}</td>
        <td>{{ '%.1f'|format(r.sieve_json[5].passing) if r.sieve_json else '-' }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty">ยังไม่มีประวัติการวิเคราะห์</div>
  {% endif %}
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
    writer.writerow(['Date', 'Time', 'AggType', 'Source', 'FM',
                     '3/4_%pass', '1/2_%pass', '3/8_%pass', '#4_%pass', '#8_%pass', 'Pan_%pass'])
    for r in rows:
        sieves = r['sieve_json'] if isinstance(r['sieve_json'], list) else _json.loads(r['sieve_json'] or '[]')
        passing = [s['passing'] for s in sieves] if sieves else [''] * 7
        # passing[0]=1", [1]=3/4", [2]=1/2", [3]=3/8", [4]=#4, [5]=#8, [6]=Pan
        writer.writerow([
            r['analyzed_at'].strftime('%Y-%m-%d'),
            r['analyzed_at'].strftime('%H:%M:%S'),
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
