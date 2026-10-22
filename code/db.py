"""
db.py — AggNet QC Database Layer (Supabase PostgreSQL)
ตาราง:
  companies  — บริษัทที่ใช้ระบบ (login account)
  analyses   — ผลการวิเคราะห์แต่ละครั้ง
"""

import psycopg2
import psycopg2.extras
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json
import os

# ── Connection ──────────────────────────────────────────────────────────────
# ใส่ password ที่ตั้งไว้ตอนสร้าง Supabase project
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:YOUR_PASSWORD_HERE@db.ybjmqizmixrqwkgreyer.supabase.co:5432/postgres"
)

def get_conn():
    return psycopg2.connect(DB_URL, connect_timeout=10)

# ── Init Tables ──────────────────────────────────────────────────────────────
INIT_SQL = """
CREATE TABLE IF NOT EXISTS companies (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(200) NOT NULL,
    username      VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analyses (
    id             SERIAL PRIMARY KEY,
    company_id     INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    analyzed_at    TIMESTAMPTZ DEFAULT NOW(),
    agg_type       VARCHAR(50),
    source         VARCHAR(200),
    fm             FLOAT,
    sieve_json     JSONB,
    zones_json     JSONB,
    image_filename VARCHAR(500)
);

CREATE INDEX IF NOT EXISTS idx_analyses_company ON analyses(company_id);
CREATE INDEX IF NOT EXISTS idx_analyses_date    ON analyses(analyzed_at DESC);
"""

def init_db():
    """สร้างตาราง (รันครั้งแรกครั้งเดียว)"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(INIT_SQL)
        conn.commit()
    print("DB: tables ready")

# ── Companies ────────────────────────────────────────────────────────────────
def create_company(name: str, username: str, password: str) -> int:
    """สร้างบัญชีบริษัทใหม่ คืน id"""
    pw_hash = generate_password_hash(password)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO companies (name, username, password_hash) VALUES (%s, %s, %s) RETURNING id",
                (name, username, pw_hash)
            )
            row = cur.fetchone()
        conn.commit()
    return row[0]

def get_company_by_username(username: str):
    """คืน dict {id, name, username, password_hash} หรือ None"""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, username, password_hash FROM companies WHERE username=%s",
                (username,)
            )
            return cur.fetchone()

def verify_login(username: str, password: str):
    """ตรวจ login คืน company dict ถ้าถูก, None ถ้าผิด"""
    company = get_company_by_username(username)
    if company and check_password_hash(company['password_hash'], password):
        return company
    return None

def list_companies():
    """รายชื่อบริษัททั้งหมด (สำหรับ admin)"""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, name, username, created_at FROM companies ORDER BY name")
            return cur.fetchall()

# ── Analyses ─────────────────────────────────────────────────────────────────
def save_analysis(company_id: int, agg_type: str, source: str,
                  fm: float, sieve_data: list, zones_data: list,
                  image_filename: str = None) -> int:
    """บันทึกผลวิเคราะห์ คืน id"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO analyses
                    (company_id, agg_type, source, fm, sieve_json, zones_json, image_filename)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                company_id, agg_type, source, fm,
                json.dumps(sieve_data),
                json.dumps(zones_data),
                image_filename
            ))
            row = cur.fetchone()
        conn.commit()
    return row[0]

def get_analyses(company_id: int, limit: int = 100,
                 agg_type: str = None, date_from: str = None, date_to: str = None):
    """ดึงประวัติของบริษัท พร้อม filter"""
    conditions = ["company_id = %s"]
    params = [company_id]

    if agg_type:
        conditions.append("agg_type = %s")
        params.append(agg_type)
    if date_from:
        conditions.append("analyzed_at >= %s")
        params.append(date_from)
    if date_to:
        conditions.append("analyzed_at <= %s")
        params.append(date_to + " 23:59:59")

    where = " AND ".join(conditions)
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, analyzed_at, agg_type, source, fm, sieve_json, zones_json, image_filename
                FROM analyses
                WHERE {where}
                ORDER BY analyzed_at DESC
                LIMIT %s
            """, params)
            return cur.fetchall()

def get_analysis_by_id(analysis_id: int, company_id: int):
    """ดึงผลวิเคราะห์ตาม id (เช็ค company_id ด้วยเพื่อความปลอดภัย)"""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM analyses WHERE id=%s AND company_id=%s",
                (analysis_id, company_id)
            )
            return cur.fetchone()

def get_fm_trend(company_id: int, agg_type: str, days: int = 30):
    """FM trend สำหรับ dashboard chart"""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT DATE(analyzed_at) as date, AVG(fm) as avg_fm,
                       MIN(fm) as min_fm, MAX(fm) as max_fm, COUNT(*) as count
                FROM analyses
                WHERE company_id=%s AND agg_type=%s
                  AND analyzed_at >= NOW() - INTERVAL '%s days'
                GROUP BY DATE(analyzed_at)
                ORDER BY date
            """, (company_id, agg_type, days))
            return cur.fetchall()
