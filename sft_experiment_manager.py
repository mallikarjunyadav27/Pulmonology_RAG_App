# sft_experiment_manager.py
"""
SFT Experiment Manager for RLHF Admin Dashboard
=================================================
Manages SFT fine-tuning experiments with LoRA on medical ranked data.
Provides DB-backed state management, background thread training,
progress tracking, and inference capabilities.

Supports dual database backend:
  - PostgreSQL (production / hosted)
  - SQLite (automatic local-dev fallback when PG is unreachable)

Integrates with pces_rlhf_experiments module via sys.path.
"""

import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

try:
    import psycopg
    _HAS_PSYCOPG = True
except ImportError:
    _HAS_PSYCOPG = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------------------------------------------------------------------
# External module import (pces_rlhf_experiments)
# ---------------------------------------------------------------------------
EXPERIMENTS_MODULE_PATH = os.getenv(
    "PCES_RLHF_EXPERIMENTS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "pces_rlhf_experiments")
)
EXPERIMENTS_MODULE_PATH = os.path.abspath(EXPERIMENTS_MODULE_PATH)

SFT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_models")
os.makedirs(SFT_MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------
db_config = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "pces_base"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5")),
}

# ---------------------------------------------------------------------------
# SQLite fallback wrapper — makes sqlite3 behave like psycopg for our usage
# ---------------------------------------------------------------------------

SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_sft.db")
_use_sqlite = False
_db_backend_initialized = False


def _adapt_sql(sql):
    """Convert PostgreSQL-dialect SQL to SQLite-compatible SQL."""
    sql = sql.replace('%s', '?')
    # Strip PostgreSQL ::type casts (must run before JSONB→TEXT replacement)
    sql = re.sub(r'::[a-zA-Z]\w*', '', sql)
    sql = re.sub(r'SERIAL\s+PRIMARY\s+KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bILIKE\b', 'LIKE', sql)
    sql = re.sub(r"DEFAULT\s+NOW\(\)", "DEFAULT CURRENT_TIMESTAMP", sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNOW\(\)', "datetime('now')", sql)
    sql = re.sub(r'\bJSONB\b', 'TEXT', sql, flags=re.IGNORECASE)
    return sql


class _SQLiteCursor:
    """Wraps sqlite3.Cursor to accept psycopg-style %s parameters."""

    def __init__(self, cursor):
        self._cur = cursor
        self._returning = False

    def execute(self, sql, params=None):
        self._returning = bool(re.search(r'RETURNING\s+\w+', sql, re.IGNORECASE))
        sql = _adapt_sql(sql)
        # Strip RETURNING clause (SQLite uses lastrowid instead)
        sql = re.sub(r'\s*RETURNING\s+\w+', '', sql, flags=re.IGNORECASE)
        if params:
            self._cur.execute(sql, tuple(params))
        else:
            self._cur.execute(sql)
        return self

    def fetchone(self):
        if self._returning:
            self._returning = False
            return (self._cur.lastrowid,)
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def lastrowid(self):
        return self._cur.lastrowid

    @property
    def rowcount(self):
        return self._cur.rowcount

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _SQLiteConn:
    """Wraps sqlite3.Connection to mimic psycopg connection interface."""

    def __init__(self, path, autocommit=False):
        if autocommit:
            self._conn = sqlite3.connect(path, isolation_level=None)
        else:
            self._conn = sqlite3.connect(path)

    def cursor(self):
        return _SQLiteCursor(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            try:
                self._conn.commit()
            except Exception:
                pass
        self._conn.close()


def _init_db_backend():
    """Detect which database backend to use (PostgreSQL or SQLite).
    
    Falls back to SQLite if:
    - SFT_USE_SQLITE=true env var is set (explicit opt-in)
    - psycopg is not installed
    - PostgreSQL is unreachable
    - Required SFT tables don't exist and can't be created (e.g. permission denied)
    """
    global _use_sqlite, _db_backend_initialized
    if _db_backend_initialized:
        return
    _db_backend_initialized = True
    # Explicit opt-in to SQLite (avoids noisy PG connection attempts in local dev)
    if os.getenv("SFT_USE_SQLITE", "").lower() in ("1", "true", "yes"):
        _use_sqlite = True
        print(f"ℹ️  SFT DB: SFT_USE_SQLITE=true, using local SQLite at {SQLITE_DB_PATH}")
        return
    if not _HAS_PSYCOPG:
        _use_sqlite = True
        print(f"⚠️  SFT DB: psycopg not installed, using local SQLite at {SQLITE_DB_PATH}")
        return
    try:
        conn = psycopg.connect(**db_config)
        with conn.cursor() as cur:
            # Check connectivity
            cur.execute("SELECT 1")
            # Check if SFT tables exist
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'sft_ranked_data'"
            )
            table_exists = cur.fetchone()[0] > 0
        if not table_exists:
            # Try to create the tables
            try:
                with conn.cursor() as cur:
                    cur.execute("""CREATE TABLE IF NOT EXISTS sft_ranked_data (
                        id SERIAL PRIMARY KEY, prompt TEXT NOT NULL,
                        response_text TEXT NOT NULL, rank INTEGER NOT NULL CHECK (rank >= 1),
                        reason TEXT, group_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW(), updated_at TIMESTAMP DEFAULT NOW(),
                        created_by INTEGER DEFAULT 1001, updated_by INTEGER DEFAULT 1001)""")
                    cur.execute("""CREATE TABLE IF NOT EXISTS sft_experiments (
                        id SERIAL PRIMARY KEY, experiment_name TEXT NOT NULL, department TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        model_name TEXT DEFAULT 'microsoft/phi-2',
                        lora_r INTEGER DEFAULT 16, lora_alpha INTEGER DEFAULT 32,
                        lora_dropout REAL DEFAULT 0.05, num_epochs INTEGER DEFAULT 10,
                        batch_size INTEGER DEFAULT 2, gradient_accumulation_steps INTEGER DEFAULT 4,
                        learning_rate REAL DEFAULT 0.0001, max_seq_length INTEGER DEFAULT 2048,
                        training_samples INTEGER DEFAULT 0, started_at TIMESTAMP,
                        completed_at TIMESTAMP, error_message TEXT, model_output_path TEXT,
                        metrics JSONB DEFAULT '{}', created_at TIMESTAMP DEFAULT NOW(),
                        created_by INTEGER DEFAULT 1001)""")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_group_id ON sft_ranked_data(group_id)")
                conn.commit()
                _use_sqlite = False
                print("✅ SFT DB: Using PostgreSQL (tables created)")
            except Exception as create_err:
                conn.close()
                _use_sqlite = True
                print(f"⚠️  SFT DB: PostgreSQL tables missing & can't create ({create_err}), using local SQLite at {SQLITE_DB_PATH}")
                return
        else:
            _use_sqlite = False
            print("✅ SFT DB: Using PostgreSQL")
        conn.close()
    except Exception as e:
        _use_sqlite = True
        print(f"⚠️  SFT DB: PostgreSQL unavailable ({type(e).__name__}: {e}), using local SQLite at {SQLITE_DB_PATH}")


def _connect(autocommit=False):
    """Get a DB connection — PostgreSQL if available, else SQLite fallback."""
    _init_db_backend()
    if _use_sqlite:
        return _SQLiteConn(SQLITE_DB_PATH, autocommit=autocommit)
    if autocommit:
        return psycopg.connect(**db_config, autocommit=True)
    return psycopg.connect(**db_config)


def _dt(val):
    """Convert datetime value to ISO string, handling both datetime and string types."""
    if val is None:
        return None
    return val.isoformat() if hasattr(val, 'isoformat') else str(val)

# ---------------------------------------------------------------------------
# Training state (in-memory, shared across threads)
# ---------------------------------------------------------------------------
_training_lock = threading.Lock()
_training_state = {
    "active": False,
    "experiment_id": None,
    "thread": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": None,
    "status": "idle",
    "log_lines": [],
    "started_at": None,
}


# ---------------------------------------------------------------------------
# Medical Departments (30) with keyword mappings for prompt classification
# ---------------------------------------------------------------------------

DEPARTMENTS = OrderedDict([
    ("Cardiology", ["heart", "cardiac", "cardiovascular", "blood pressure", "hypertension", "cholesterol", "atherosclerosis", "heart attack", "arrhythmia", "coronary", "angina"]),
    ("Neurology", ["brain", "nerve", "neurological", "stroke", "migraine", "headache", "seizure", "epilepsy", "neuropathy", "parkinson", "alzheimer"]),
    ("Diabetes", ["diabetes", "diabetic", "insulin", "blood sugar", "glucose", "glycemic", "hba1c", "type-1 diabetes", "type-2 diabetes", "hyperglycemia", "hypoglycemia"]),
    ("Pulmonology", ["lung", "respiratory", "asthma", "pneumonia", "breathing", "bronchitis", "copd", "pulmonary", "bronchial"]),
    ("Gastroenterology", ["stomach", "digestive", "gastric", "intestin", "bowel", "colon", "acid reflux", "gerd", "gastrointestinal", "celiac"]),
    ("Nephrology", ["kidney", "renal", "nephro", "dialysis", "urinary tract infection", "uti"]),
    ("Oncology", ["cancer", "tumor", "malignant", "chemotherapy", "radiation therapy", "oncolog", "carcinoma", "leukemia", "lymphoma"]),
    ("Hematology", ["blood cell", "anemia", "hemoglobin", "platelet", "coagulation", "iron deficiency", "white blood cell", "red blood cell", "blood clot"]),
    ("Orthopedics", ["bone", "joint", "fracture", "sprain", "osteoporosis", "orthopedic", "musculoskeletal", "tendon", "ligament"]),
    ("Dermatology", ["skin", "rash", "eczema", "psoriasis", "acne", "melanoma", "dermatitis", "dermatolog"]),
    ("Ophthalmology", ["eye", "vision", "optic", "glaucoma", "cataract", "retina", "ophthalmolog"]),
    ("Psychiatry", ["mental", "depression", "anxiety", "psychiatric", "psycholog", "bipolar", "schizophrenia", "mood disorder", "ptsd"]),
    ("Pediatrics", ["child", "infant", "pediatric", "newborn", "neonatal", "adolescent"]),
    ("Rheumatology", ["autoimmune", "rheumat", "lupus", "fibromyalgia", "inflammatory"]),
    ("Urology", ["urinary", "bladder", "prostate", "urology", "urolog", "kidney stone"]),
    ("Immunology & Allergy", ["allergy", "allergic", "immune system", "immunology", "vaccine", "vaccination", "anaphylaxis", "histamine"]),
    ("Infectious Disease", ["infection", "bacterial", "viral", "antibiotic", "virus", "bacteria", "pathogen", "sepsis", "contagious"]),
    ("Emergency Medicine", ["emergency", "trauma", "first aid", "cpr", "resuscitation", "acute care", "critical care"]),
    ("Hepatology", ["liver", "hepat", "cirrhosis", "hepatitis", "fatty liver", "alcohol"]),
    ("Nutrition & Dietetics", ["diet", "nutrition", "vitamin", "mineral", "supplement", "mediterranean", "calorie", "obesity", "hydration", "water intake"]),
    ("Pharmacology", ["drug", "medication", "pharmaceutical", "side effect", "ibuprofen", "dosage", "contraindication", "prescription"]),
    ("General Medicine", ["general health", "wellness", "physical exam", "checkup", "primary care", "symptom", "diagnosis"]),
    ("ENT (Otolaryngology)", ["ear", "nose", "throat", "sinus", "hearing", "tinnitus", "larynx", "pharynx"]),
    ("Obstetrics & Gynecology", ["pregnancy", "prenatal", "obstetric", "gynecolog", "menstrual", "ovarian", "uterine", "fertility"]),
    ("Radiology", ["x-ray", "mri", "ct scan", "ultrasound", "radiology", "imaging", "mammogram"]),
    ("Anesthesiology", ["anesthes", "sedation", "pain relief", "local anesthetic", "general anesthetic"]),
    ("Pain Management", ["chronic pain", "analgesic", "nerve block", "pain management", "opioid"]),
    ("Sleep Medicine", ["sleep", "insomnia", "sleep apnea", "circadian", "sleep deprivation", "melatonin"]),
    ("Sports Medicine", ["sports", "exercise", "physical activity", "athletic", "fitness", "physical exercise"]),
    ("Geriatrics", ["elderly", "aging", "geriatric", "dementia", "age-related", "senior"]),
])


# ============================================================================
# Table Management
# ============================================================================

def ensure_tables():
    """Create sft_ranked_data, sft_experiments, and sme_doctors tables if they don't exist.
    Also detects the database backend (PostgreSQL or SQLite).
    """
    _init_db_backend()
    create_ranked_data = """
    CREATE TABLE IF NOT EXISTS sft_ranked_data (
        id SERIAL PRIMARY KEY,
        prompt TEXT NOT NULL,
        response_text TEXT NOT NULL,
        rank INTEGER NOT NULL CHECK (rank >= 1),
        reason TEXT,
        group_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        created_by INTEGER DEFAULT 1001,
        updated_by INTEGER DEFAULT 1001
    );
    """
    create_experiments = """
    CREATE TABLE IF NOT EXISTS sft_experiments (
        id SERIAL PRIMARY KEY,
        experiment_name TEXT NOT NULL,
        department TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        model_name TEXT DEFAULT 'microsoft/phi-2',
        lora_r INTEGER DEFAULT 16,
        lora_alpha INTEGER DEFAULT 32,
        lora_dropout REAL DEFAULT 0.05,
        num_epochs INTEGER DEFAULT 10,
        batch_size INTEGER DEFAULT 2,
        gradient_accumulation_steps INTEGER DEFAULT 4,
        learning_rate REAL DEFAULT 0.0001,
        max_seq_length INTEGER DEFAULT 2048,
        training_samples INTEGER DEFAULT 0,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        error_message TEXT,
        model_output_path TEXT,
        metrics JSONB DEFAULT '{}',
        created_at TIMESTAMP DEFAULT NOW(),
        created_by INTEGER DEFAULT 1001
    );
    """
    create_doctors = """
    CREATE TABLE IF NOT EXISTS sme_doctors (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        department TEXT NOT NULL,
        specialty TEXT,
        is_active BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """
    # Add department column to existing tables (safe ALTER — ignored if column exists)
    add_department_col = """
    ALTER TABLE sft_experiments ADD COLUMN department TEXT;
    """
    create_index = """
    CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_group_id ON sft_ranked_data(group_id);
    """
    create_doctors_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_sme_doctors_department ON sme_doctors(department);",
        "CREATE INDEX IF NOT EXISTS idx_sme_doctors_active ON sme_doctors(is_active, department);",
        "CREATE INDEX IF NOT EXISTS idx_sme_doctors_name ON sme_doctors(name);",
    ]
    try:
        with _connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(create_ranked_data)
                cur.execute(create_experiments)
                cur.execute(create_doctors)
                cur.execute(create_index)
                for idx_sql in create_doctors_indexes:
                    try:
                        cur.execute(idx_sql)
                    except Exception:
                        pass  # Index already exists
                try:
                    cur.execute(add_department_col)
                except Exception:
                    pass  # Column already exists
        print("✅ SFT experiment tables ensured")
        return True
    except Exception as e:
        print(f"❌ Error creating SFT tables: {e}")
        return False


# ============================================================================
# Ranked Data CRUD
# ============================================================================

def get_ranked_data(group_id=None, search=None, page=1, per_page=50, sme_filter=None, domain=None, reason_empty=False):
    """Retrieve ranked data, optionally filtered by group, search, domain, or SME score.
    
    Args:
        sme_filter: None=all, 'reviewed'=only SME-scored, 'pending'=not yet scored, 'high'=score>=4
        domain: Department name to filter by (matches the domain column)
        reason_empty: If True, only return entries where reason is NULL or empty
    """
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                conditions = []
                params = []

                if group_id:
                    conditions.append("group_id = %s")
                    params.append(group_id)
                if search:
                    conditions.append("(prompt ILIKE %s OR response_text ILIKE %s)")
                    params.extend([f"%{search}%", f"%{search}%"])
                if domain:
                    conditions.append("LOWER(domain) = LOWER(%s)")
                    params.append(domain)
                if reason_empty:
                    conditions.append("(reason IS NULL OR reason = '')")
                if sme_filter == 'reviewed':
                    conditions.append("sme_score IS NOT NULL")
                elif sme_filter == 'pending':
                    conditions.append("sme_score IS NULL")
                elif sme_filter == 'high':
                    conditions.append("sme_score >= 4")

                where = "WHERE " + " AND ".join(conditions) if conditions else ""
                offset = (page - 1) * per_page

                cur.execute(
                    f"SELECT COUNT(DISTINCT group_id) FROM sft_ranked_data {where}",
                    params,
                )
                total_groups = cur.fetchone()[0]

                cur.execute(
                    f"""SELECT id, prompt, response_text, rank, reason, group_id,
                               created_at, updated_at, domain,
                               sme_score, sme_score_reason, sme_reviewed_by, sme_reviewed_at
                        FROM sft_ranked_data {where}
                        ORDER BY group_id, rank
                        LIMIT %s OFFSET %s""",
                    params + [per_page * 3, offset * 3],
                )
                rows = cur.fetchall()

                data = []
                for row in rows:
                    data.append({
                        "id": row[0],
                        "prompt": row[1],
                        "response_text": row[2],
                        "rank": row[3],
                        "reason": row[4],
                        "group_id": row[5],
                        "created_at": _dt(row[6]),
                        "updated_at": _dt(row[7]),
                        "domain": row[8],
                        "sme_score": row[9],
                        "sme_score_reason": row[10],
                        "sme_reviewed_by": row[11],
                        "sme_reviewed_at": _dt(row[12]),
                    })

                return {
                    "success": True,
                    "data": data,
                    "total_groups": total_groups,
                    "page": page,
                    "per_page": per_page,
                }
    except Exception as e:
        return {"success": False, "error": str(e)}


def add_ranked_entry(prompt, responses):
    """Add a new ranked data group (prompt + multiple ranked responses).

    Args:
        prompt: The medical question
        responses: List of dicts with keys: text, rank, reason
    """
    group_id = str(uuid.uuid4())[:8]
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                for resp in responses:
                    cur.execute(
                        """INSERT INTO sft_ranked_data
                           (prompt, response_text, rank, reason, group_id)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (prompt, resp["text"], resp["rank"], resp.get("reason", ""), group_id),
                    )
            conn.commit()
        return {"success": True, "group_id": group_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


def update_ranked_entry(entry_id, response_text=None, rank=None, reason=None):
    """Update a single ranked data entry."""
    try:
        updates = []
        params = []
        if response_text is not None:
            updates.append("response_text = %s")
            params.append(response_text)
        if rank is not None:
            updates.append("rank = %s")
            params.append(rank)
        if reason is not None:
            updates.append("reason = %s")
            params.append(reason)

        if not updates:
            return {"success": False, "error": "No fields to update"}

        updates.append("updated_at = NOW()")
        params.append(entry_id)

        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE sft_ranked_data SET {', '.join(updates)} WHERE id = %s",
                    params,
                )
            conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_ranked_entry(entry_id):
    """Delete a single ranked data entry."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sft_ranked_data WHERE id = %s", (entry_id,))
            conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_ranked_group(group_id):
    """Delete all entries in a ranked data group."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sft_ranked_data WHERE group_id = %s", (group_id,))
            conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_ranked_data_stats():
    """Get summary statistics about ranked training data."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM sft_ranked_data")
                total_entries = cur.fetchone()[0]

                cur.execute("SELECT COUNT(DISTINCT group_id) FROM sft_ranked_data")
                total_groups = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM sft_ranked_data WHERE rank = 1")
                rank1_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM sft_ranked_data WHERE sme_score IS NOT NULL")
                sme_reviewed_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM sft_ranked_data WHERE sme_score >= 4")
                sme_high_score_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(DISTINCT group_id) FROM sft_ranked_data WHERE sme_score IS NOT NULL")
                sme_reviewed_groups = cur.fetchone()[0]

                return {
                    "success": True,
                    "total_entries": total_entries,
                    "total_groups": total_groups,
                    "rank1_count": rank1_count,
                    "sme_reviewed_count": sme_reviewed_count,
                    "sme_high_score_count": sme_high_score_count,
                    "sme_reviewed_groups": sme_reviewed_groups,
                }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# JSONL Import / Export
# ============================================================================

def import_from_jsonl(file_path=None):
    """Import ranked data from a JSONL file into PostgreSQL.

    Args:
        file_path: Path to JSONL file. Defaults to the external module's file.
    """
    if file_path is None:
        file_path = os.path.join(EXPERIMENTS_MODULE_PATH, "medical_ranked.jsonl")

    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    imported = 0
    skipped = 0
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            records = []
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
                    continue

        with _connect() as conn:
            with conn.cursor() as cur:
                for record in records:
                    prompt = record.get("prompt", "")
                    responses = record.get("responses", [])
                    group_id = str(uuid.uuid4())[:8]

                    # Check for duplicate prompt
                    cur.execute(
                        "SELECT COUNT(*) FROM sft_ranked_data WHERE prompt = %s",
                        (prompt,),
                    )
                    if cur.fetchone()[0] > 0:
                        skipped += 1
                        continue

                    for resp in responses:
                        cur.execute(
                            """INSERT INTO sft_ranked_data
                               (prompt, response_text, rank, reason, group_id)
                               VALUES (%s, %s, %s, %s, %s)""",
                            (
                                prompt,
                                resp.get("text", ""),
                                resp.get("rank", 0),
                                resp.get("reason", ""),
                                group_id,
                            ),
                        )
                        imported += 1
            conn.commit()

        return {
            "success": True,
            "imported": imported,
            "skipped": skipped,
            "file": file_path,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def export_to_jsonl():
    """Export ranked data from PostgreSQL to JSONL format (returned as string)."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT DISTINCT group_id FROM sft_ranked_data ORDER BY group_id"""
                )
                group_ids = [row[0] for row in cur.fetchall()]

                lines = []
                for gid in group_ids:
                    cur.execute(
                        """SELECT prompt, response_text, rank, reason
                           FROM sft_ranked_data
                           WHERE group_id = %s ORDER BY rank""",
                        (gid,),
                    )
                    rows = cur.fetchall()
                    if not rows:
                        continue

                    record = {
                        "prompt": rows[0][0],
                        "responses": [
                            {"text": r[1], "rank": r[2], "reason": r[3]}
                            for r in rows
                        ],
                    }
                    lines.append(json.dumps(record, ensure_ascii=False))

                return {"success": True, "data": "\n".join(lines), "count": len(lines)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Experiment CRUD
# ============================================================================

def recover_stuck_experiments():
    """Auto-recover experiments stuck as 'running' when model dir exists on disk."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, experiment_name, model_output_path, department FROM sft_experiments WHERE status = %s",
                    ("running",),
                )
                stuck = cur.fetchall()
        recovered = 0
        for row in stuck:
            exp_id, name, saved_path, dept = row[0], row[1], row[2], row[3]
            # Try saved path first, then derive from naming convention
            candidates = [saved_path] if saved_path else []
            safe_dept = re.sub(r'[^a-zA-Z0-9_-]', '_', dept) if dept else None
            if safe_dept:
                candidates.append(os.path.join("sft_models", f"{safe_dept}_experiment_{exp_id}"))
            candidates.append(os.path.join("sft_models", f"experiment_{exp_id}"))

            for path in candidates:
                if path and os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                    _update_experiment_status(
                        exp_id,
                        status="completed",
                        completed_at=datetime.utcnow(),
                        model_output_path=path,
                    )
                    print(f"✅ Auto-recovered experiment {exp_id} ({name}) — model found at {path}")
                    recovered += 1
                    break
        return recovered
    except Exception as e:
        print(f"Error in recover_stuck_experiments: {e}")
        return 0

def list_experiments(page=1, per_page=20):
    """List all SFT experiments with pagination."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM sft_experiments")
                total = cur.fetchone()[0]

                offset = (page - 1) * per_page
                cur.execute(
                    """SELECT id, experiment_name, status, model_name,
                              num_epochs, learning_rate, training_samples,
                              started_at, completed_at, error_message,
                              model_output_path, metrics, created_at, lora_r, lora_alpha,
                              department
                       FROM sft_experiments
                       ORDER BY created_at DESC
                       LIMIT %s OFFSET %s""",
                    (per_page, offset),
                )
                rows = cur.fetchall()

                experiments = []
                for row in rows:
                    experiments.append({
                        "id": row[0],
                        "experiment_name": row[1],
                        "status": row[2],
                        "model_name": row[3],
                        "num_epochs": row[4],
                        "learning_rate": row[5],
                        "training_samples": row[6],
                        "started_at": _dt(row[7]),
                        "completed_at": _dt(row[8]),
                        "error_message": row[9],
                        "model_output_path": row[10],
                        "metrics": row[11] or {},
                        "created_at": _dt(row[12]),
                        "lora_r": row[13],
                        "lora_alpha": row[14],
                        "department": row[15],
                    })

                return {"success": True, "experiments": experiments, "total": total}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_experiment(experiment_id):
    """Get details of a single experiment."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, experiment_name, status, model_name,
                              lora_r, lora_alpha, lora_dropout, num_epochs,
                              batch_size, gradient_accumulation_steps,
                              learning_rate, max_seq_length, training_samples,
                              started_at, completed_at, error_message,
                              model_output_path, metrics, created_at
                       FROM sft_experiments WHERE id = %s""",
                    (experiment_id,),
                )
                row = cur.fetchone()
                if not row:
                    return {"success": False, "error": "Experiment not found"}

                return {
                    "success": True,
                    "experiment": {
                        "id": row[0],
                        "experiment_name": row[1],
                        "status": row[2],
                        "model_name": row[3],
                        "lora_r": row[4],
                        "lora_alpha": row[5],
                        "lora_dropout": row[6],
                        "num_epochs": row[7],
                        "batch_size": row[8],
                        "gradient_accumulation_steps": row[9],
                        "learning_rate": row[10],
                        "max_seq_length": row[11],
                        "training_samples": row[12],
                        "started_at": _dt(row[13]),
                        "completed_at": _dt(row[14]),
                        "error_message": row[15],
                        "model_output_path": row[16],
                        "metrics": row[17] or {},
                        "created_at": _dt(row[18]),
                    },
                }
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_experiment(experiment_id):
    """Delete an experiment record."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM sft_experiments WHERE id = %s", (experiment_id,)
                )
            conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def update_experiment_samples(experiment_id):
    """Recalculate and persist training_samples for an experiment using the live domain-aware count.

    Uses get_prompts_by_department (domain column first, keyword fallback) so the count
    always reflects the full prompt set for the experiment's department.
    """
    try:
        exp_result = get_experiment(experiment_id)
        if not exp_result.get("success"):
            return exp_result
        department = exp_result["experiment"].get("department")
        if not department:
            return {"success": False, "error": "Experiment has no department set"}
        count_result = get_prompts_by_department(department, limit=10000)
        if not count_result.get("success"):
            return count_result
        new_count = count_result["total"]
        _update_experiment_status(experiment_id, training_samples=new_count)
        return {"success": True, "training_samples": new_count, "department": department}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Training Orchestration (Background Thread)
# ============================================================================

def _build_training_data_from_db(department=None, use_sme_scores=False, min_sme_score=1):
    """Build training dataset from sft_ranked_data.

    Args:
        department: If provided, only include prompts matching this department's keywords.
        use_sme_scores: If True, use SME score-based weighting instead of rank-based selection.
        min_sme_score: Minimum SME score required (1-5). Only used when use_sme_scores=True.

    Returns list of dicts matching the format expected by train_medical_sft.
    When use_sme_scores=True, samples are weighted by duplicating records based on their score.
    """
    records = []
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                # Build base query conditions
                conditions = []
                params = []

                if department and department in DEPARTMENTS:
                    keywords = DEPARTMENTS[department]
                    kw_conditions = " OR ".join(["LOWER(prompt) LIKE %s" for _ in keywords])
                    # Match rows tagged with this domain OR untagged rows that hit a keyword
                    conditions.append(
                        f"(LOWER(domain) = LOWER(%s) OR "
                        f"((domain IS NULL OR TRIM(domain) = '') AND ({kw_conditions})))"
                    )
                    params.extend([department] + [f"%{kw.lower()}%" for kw in keywords])
                
                # SME score-based filtering
                if use_sme_scores:
                    conditions.append("sme_score IS NOT NULL")
                    conditions.append("sme_score >= %s")
                    params.append(min_sme_score)
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                # Get distinct group_ids
                cur.execute(
                    f"SELECT DISTINCT group_id FROM sft_ranked_data WHERE {where_clause} ORDER BY group_id",
                    params,
                )
                group_ids = [r[0] for r in cur.fetchall()]

                for gid in group_ids:
                    if use_sme_scores:
                        # Get records with SME scores for weighting
                        cur.execute(
                            """SELECT prompt, response_text, rank, reason, sme_score, sme_score_reason
                               FROM sft_ranked_data
                               WHERE group_id = %s AND sme_score IS NOT NULL AND sme_score >= %s
                               ORDER BY sme_score DESC, rank""",
                            (gid, min_sme_score),
                        )
                    else:
                        cur.execute(
                            """SELECT prompt, response_text, rank, reason
                               FROM sft_ranked_data
                               WHERE group_id = %s ORDER BY rank""",
                            (gid,),
                        )
                    rows = cur.fetchall()
                    if not rows:
                        continue
                    
                    if use_sme_scores:
                        # Score-based weighting: duplicate records based on SME score
                        # Score 5 = 5 copies, Score 4 = 4 copies, etc.
                        for r in rows:
                            sme_score = r[4] or 1
                            sme_reason = r[5] or r[3]  # Use SME reason if available, else original reason
                            base_record = {
                                "prompt": r[0],
                                "responses": [
                                    {"text": r[1], "rank": r[2], "reason": sme_reason, "sme_score": sme_score}
                                ],
                            }
                            # Add weighted copies (score determines weight)
                            for _ in range(sme_score):
                                records.append(base_record.copy())
                    else:
                        # Original rank-based selection (only rank-1)
                        record = {
                            "prompt": rows[0][0],
                            "responses": [
                                {"text": r[1], "rank": r[2], "reason": r[3]}
                                for r in rows
                            ],
                        }
                        records.append(record)
    except Exception as e:
        print(f"Error building training data from DB: {e}")
    return records


def _write_temp_jsonl(records):
    """Write records to a temporary JSONL file for the training script."""
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="sft_training_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _update_experiment_status(experiment_id, **kwargs):
    """Update experiment fields in the database with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            fields = []
            params = []
            for key, val in kwargs.items():
                if key == "metrics" and isinstance(val, dict):
                    fields.append(f"{key} = %s::jsonb")
                    params.append(json.dumps(val))
                elif key in ("completed_at", "started_at") and val is not None:
                    fields.append(f"{key} = %s")
                    params.append(val.isoformat() if hasattr(val, 'isoformat') else str(val))
                else:
                    fields.append(f"{key} = %s")
                    params.append(val)
            params.append(experiment_id)

            with _connect(autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE sft_experiments SET {', '.join(fields)} WHERE id = %s",
                        params,
                    )
            return  # success
        except Exception as e:
            print(f"Error updating experiment {experiment_id} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                # Log to training state so it appears in the UI
                if _training_state.get("active"):
                    _training_state["log_lines"].append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ DB update failed after {max_retries} retries: {e}"
                    )


def _run_training(experiment_id, config, department=None, use_sme_scores=False, min_sme_score=1):
    """Background training function executed in a separate thread.

    Imports train_medical_sft from the external module and runs training
    with the given configuration. Updates experiment status as it progresses.

    Args:
        experiment_id: DB experiment record ID
        config: Training hyperparameters
        department: If provided, only train on this department's data
        use_sme_scores: If True, use SME score-based weighting
        min_sme_score: Minimum SME score required (1-5)
    """
    global _training_state

    try:
        dept_label = f" [{department}]" if department else ""
        score_label = f" (SME weighted, min={min_sme_score})" if use_sme_scores else ""
        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting experiment {experiment_id}{dept_label}{score_label}")
        _update_experiment_status(experiment_id, status="running", started_at=datetime.utcnow())

        # Build dataset from DB (filtered by department and/or SME scores)
        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading training data from database{dept_label}...")
        records = _build_training_data_from_db(department=department, use_sme_scores=use_sme_scores, min_sme_score=min_sme_score)
        if not records:
            dept_msg = f" for department '{department}'" if department else ""
            score_msg = f" with SME score >= {min_sme_score}" if use_sme_scores else ""
            raise ValueError(f"No training data found in database{dept_msg}{score_msg}. Import data first or lower the minimum score.")

        if use_sme_scores:
            _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(records)} weighted training samples (score-based replication)")
        else:
            _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(records)} training groups")
        _update_experiment_status(experiment_id, training_samples=len(records))

        # Write temp JSONL for the training script
        temp_jsonl = _write_temp_jsonl(records)
        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Prepared temporary training file")

        # Set up output directory (include department in path for scoped models)
        if department:
            safe_dept = re.sub(r'[^a-zA-Z0-9_-]', '_', department)
            output_dir = os.path.join(SFT_MODELS_DIR, f"{safe_dept}_experiment_{experiment_id}")
        else:
            output_dir = os.path.join(SFT_MODELS_DIR, f"experiment_{experiment_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Import training functions from external module
        if EXPERIMENTS_MODULE_PATH not in sys.path:
            sys.path.insert(0, EXPERIMENTS_MODULE_PATH)

        import importlib
        # Force reimport to pick up any changes
        if "train_medical_sft" in sys.modules:
            importlib.reload(sys.modules["train_medical_sft"])
        import train_medical_sft as sft_module

        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model: {config['model_name']}")

        # Override module-level config
        sft_module.MODEL_NAME = config["model_name"]
        sft_module.DATA_FILE = temp_jsonl
        sft_module.OUTPUT_DIR = output_dir
        sft_module.NUM_EPOCHS = config["num_epochs"]
        sft_module.BATCH_SIZE = config["batch_size"]
        sft_module.GRADIENT_ACCUMULATION_STEPS = config["gradient_accumulation_steps"]
        sft_module.LEARNING_RATE = config["learning_rate"]
        sft_module.LORA_R = config["lora_r"]
        sft_module.LORA_ALPHA = config["lora_alpha"]
        sft_module.LORA_DROPOUT = config.get("lora_dropout", 0.05)
        sft_module.MAX_SEQ_LENGTH = config.get("max_seq_length", 2048)

        # Build dataset
        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Building HuggingFace dataset...")
        dataset = sft_module.build_dataset(temp_jsonl)
        _training_state["total_epochs"] = config["num_epochs"]

        # Load model and tokenizer
        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model and tokenizer...")
        model, tokenizer = sft_module.load_model_and_tokenizer(config["model_name"])

        # Apply LoRA
        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Applying LoRA adapter (r={config['lora_r']}, alpha={config['lora_alpha']})...")
        model = sft_module.apply_lora(model)

        # Install a custom callback to track progress
        from transformers import TrainerCallback

        class ProgressCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                _training_state["current_epoch"] = int(state.epoch)
                _training_state["log_lines"].append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {int(state.epoch)}/{config['num_epochs']} completed"
                )

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    _training_state["current_loss"] = logs["loss"]
                    _training_state["log_lines"].append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Step {state.global_step}: loss={logs['loss']:.4f}"
                    )

        # Run training with custom callback
        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting SFT training...")

        from trl import SFTConfig, SFTTrainer

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            fp16=sft_module.USE_FP16,
            bf16=sft_module.USE_BF16,
            save_strategy="epoch",
            logging_steps=1,
            report_to="none",
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            max_length=config.get("max_seq_length", 2048),
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_config,
            callbacks=[ProgressCallback()],
        )

        train_result = trainer.train()

        # Save model
        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Collect metrics
        metrics = {}
        if train_result and hasattr(train_result, "metrics"):
            metrics = {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in train_result.metrics.items()
            }

        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Training completed successfully!")

        _update_experiment_status(
            experiment_id,
            status="completed",
            completed_at=datetime.utcnow(),
            model_output_path=output_dir,
            metrics=metrics,
        )

        # Clean up temp file
        try:
            os.remove(temp_jsonl)
        except OSError:
            pass

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        _training_state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Training failed: {str(e)}")
        _update_experiment_status(
            experiment_id,
            status="failed",
            completed_at=datetime.utcnow(),
            error_message=error_msg,
        )
    finally:
        with _training_lock:
            _training_state["active"] = False
            _training_state["status"] = "idle"
            _training_state["thread"] = None


def start_experiment(name, config, department=None, use_sme_scores=False, min_sme_score=1):
    """Create a new experiment and start training in a background thread.

    Args:
        name: Human-readable experiment name
        config: Dict with training hyperparameters
        department: If provided, only train on this department's data
        use_sme_scores: If True, use SME score-based weighting instead of rank-based selection
        min_sme_score: Minimum SME score required (1-5). Only used when use_sme_scores=True.

    Returns:
        Dict with success status and experiment_id
    """
    global _training_state

    with _training_lock:
        if _training_state["active"]:
            return {
                "success": False,
                "error": "A training experiment is already running. Wait for it to complete.",
            }

    # Validate training data exists (scoped to department if provided)
    if department and department in DEPARTMENTS:
        dept_data = get_prompts_by_department(department, limit=1000)
        sample_count = dept_data.get("total", 0)
        if sample_count == 0:
            return {
                "success": False,
                "error": f"No training data available for department '{department}'.",
            }
    else:
        stats = get_ranked_data_stats()
        if not stats.get("success") or stats.get("rank1_count", 0) == 0:
            return {
                "success": False,
                "error": "No training data available. Import ranked data first.",
            }

    defaults = {
        "model_name": "microsoft/phi-2",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "num_epochs": 10,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "max_seq_length": 2048,
    }
    merged_config = {**defaults, **config}

    # Create experiment record in DB
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO sft_experiments
                       (experiment_name, department, status, model_name, lora_r, lora_alpha,
                        lora_dropout, num_epochs, batch_size,
                        gradient_accumulation_steps, learning_rate, max_seq_length)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (
                        name,
                        department,
                        "pending",
                        merged_config["model_name"],
                        merged_config["lora_r"],
                        merged_config["lora_alpha"],
                        merged_config["lora_dropout"],
                        merged_config["num_epochs"],
                        merged_config["batch_size"],
                        merged_config["gradient_accumulation_steps"],
                        merged_config["learning_rate"],
                        merged_config["max_seq_length"],
                    ),
                )
                experiment_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        return {"success": False, "error": f"Failed to create experiment: {str(e)}"}

    # Start background thread
    with _training_lock:
        _training_state = {
            "active": True,
            "experiment_id": experiment_id,
            "thread": None,
            "current_epoch": 0,
            "total_epochs": merged_config["num_epochs"],
            "current_loss": None,
            "status": "starting",
            "log_lines": [],
            "started_at": datetime.utcnow().isoformat(),
        }

        t = threading.Thread(
            target=_run_training,
            args=(experiment_id, merged_config, department, use_sme_scores, min_sme_score),
            daemon=True,
        )
        _training_state["thread"] = t
        t.start()

    return {"success": True, "experiment_id": experiment_id, "use_sme_scores": use_sme_scores, "min_sme_score": min_sme_score}


def get_training_status():
    """Get the current training status (for polling from the UI)."""
    return {
        "active": _training_state["active"],
        "experiment_id": _training_state["experiment_id"],
        "current_epoch": _training_state["current_epoch"],
        "total_epochs": _training_state["total_epochs"],
        "current_loss": _training_state["current_loss"],
        "status": _training_state["status"],
        "log_lines": _training_state["log_lines"][-50:],  # Last 50 lines
        "started_at": _training_state["started_at"],
    }


# ============================================================================
# Model Inference (Test trained model)
# ============================================================================

def test_trained_model(experiment_id, question, max_new_tokens=256):
    """Test a trained model from a completed experiment.

    Args:
        experiment_id: The experiment ID whose model to load
        question: Medical question to ask
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dict with generated response and metadata
    """
    exp = get_experiment(experiment_id)
    if not exp.get("success"):
        return {"success": False, "error": "Experiment not found"}

    experiment = exp["experiment"]
    if experiment["status"] != "completed":
        return {"success": False, "error": f"Experiment status is '{experiment['status']}', not 'completed'"}

    model_path = experiment.get("model_output_path")
    if not model_path or not os.path.exists(model_path):
        return {"success": False, "error": f"Model path not found: {model_path}"}

    try:
        # Import inference module
        if EXPERIMENTS_MODULE_PATH not in sys.path:
            sys.path.insert(0, EXPERIMENTS_MODULE_PATH)

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_time = time.time()

        # Load tokenizer from the trained model path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine device and precision
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.bfloat16
        else:
            device = "cpu"
            torch_dtype = torch.float32

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            experiment["model_name"],
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )

        # Load and merge LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()

        if device != "cuda":
            model = model.to(device)

        model.eval()
        model.config.use_cache = True

        # Generate response
        system_prompt = (
            "You are a medical doctor providing clinical consultation.\n"
            "Answer clearly and concisely in plain prose.\n"
            "Do NOT include Python code, code examples, programming exercises, or tutorial content.\n"
            "Do NOT include markdown headers or bullet lists unless listing medications.\n"
            "Provide evidence-based clinical guidance appropriate for a healthcare professional.\n"
        )
        prompt = (
            f"### System:\n{system_prompt}\n\n"
            f"### User:\n{question}\n\n"
            f"### Assistant:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Strip code blocks, exercise sections, and tutorial artifacts
        import re as _re
        response = _re.sub(r'```[\s\S]*?```', '', response)
        response = _re.sub(r'`[^`]+`', '', response)
        response = _re.sub(
            r'\n*(#+\s*)?(Exercise|Ideas?:|Solution:|Example[:\s]|import |def |print\()[^\n]*(\n[^\n]+)*',
            '', response, flags=_re.IGNORECASE
        )
        response = _re.sub(r'\n{3,}', '\n\n', response).strip()

        elapsed = time.time() - start_time
        tokens_generated = len(output_ids[0]) - inputs["input_ids"].shape[1]

        return {
            "success": True,
            "response": response,
            "question": question,
            "experiment_id": experiment_id,
            "experiment_name": experiment["experiment_name"],
            "model_name": experiment["model_name"],
            "device": device,
            "tokens_generated": int(tokens_generated),
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ============================================================================
# Department-based Queries
# ============================================================================

def get_department_list():
    """Get list of all medical departments."""
    return {
        "success": True,
        "departments": list(DEPARTMENTS.keys()),
    }


def get_prompts_by_department(department, limit=10, reason_empty_only=False):
    """Get prompts matching a department using domain column first, then keyword fallback.

    Args:
        department: Department name from DEPARTMENTS dict
        limit: Max number of prompt groups to return (default 10)
        reason_empty_only: If True, only return groups with ≥1 empty reason
    """
    keywords = DEPARTMENTS.get(department, [])
    if not keywords:
        return {"success": False, "error": f"Unknown department: {department}"}

    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                reason_filter = """
                    AND group_id IN (
                        SELECT group_id FROM sft_ranked_data
                        WHERE reason IS NULL OR TRIM(reason) = ''
                    )
                """ if reason_empty_only else ""

                # Primary: match rows explicitly tagged with this domain
                domain_sql = f"""
                    SELECT DISTINCT group_id FROM sft_ranked_data
                    WHERE LOWER(domain) = LOWER(%s)
                    {reason_filter}
                    LIMIT %s
                """
                cur.execute(domain_sql, [department, limit])
                group_ids = [row[0] for row in cur.fetchall()]

                # Fallback: keyword match for rows without a domain tag
                if len(group_ids) < limit:
                    kw_conditions = " OR ".join(["LOWER(prompt) LIKE %s" for _ in keywords])
                    kw_params = [f"%{kw.lower()}%" for kw in keywords]
                    exclude = ", ".join(["%s"] * len(group_ids)) if group_ids else "NULL"
                    exclude_clause = f"AND group_id NOT IN ({exclude})" if group_ids else ""
                    remaining = limit - len(group_ids)
                    kw_sql = f"""
                        SELECT DISTINCT group_id FROM sft_ranked_data
                        WHERE (domain IS NULL OR TRIM(domain) = '')
                        AND ({kw_conditions})
                        {exclude_clause}
                        {reason_filter}
                        LIMIT %s
                    """
                    cur.execute(kw_sql, kw_params + group_ids + [remaining])
                    group_ids += [row[0] for row in cur.fetchall()]

                if not group_ids:
                    return {
                        "success": True,
                        "data": [],
                        "total": 0,
                        "department": department,
                    }

                # Get all entries for matched groups
                placeholders = ", ".join(["%s"] * len(group_ids))
                cur.execute(
                    f"""SELECT id, prompt, response_text, rank, reason, group_id, created_at
                        FROM sft_ranked_data
                        WHERE group_id IN ({placeholders})
                        ORDER BY group_id, rank""",
                    group_ids,
                )
                rows = cur.fetchall()

                data = []
                for row in rows:
                    data.append({
                        "id": row[0],
                        "prompt": row[1],
                        "response_text": row[2],
                        "rank": row[3],
                        "reason": row[4],
                        "group_id": row[5],
                        "created_at": _dt(row[6]),
                    })

                return {
                    "success": True,
                    "data": data,
                    "total": len(group_ids),
                    "department": department,
                }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Doctor Management (SME Doctors)
# ============================================================================

def get_doctors(department=None, active_only=True):
    """Get list of doctors, optionally filtered by department.
    
    Args:
        department: If provided, only return doctors from this department
        active_only: If True (default), only return active doctors
    
    Returns:
        dict with success status and list of doctors
    """
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                query = "SELECT id, name, email, department, specialty, is_active, created_at, updated_at FROM sme_doctors WHERE 1=1"
                params = []
                
                if active_only:
                    query += " AND is_active = %s"
                    params.append(True if not _use_sqlite else 1)
                
                if department:
                    query += " AND department = %s"
                    params.append(department)
                
                query += " ORDER BY department, name"
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                doctors = []
                for row in rows:
                    doctors.append({
                        "id": row[0],
                        "name": row[1],
                        "email": row[2],
                        "department": row[3],
                        "specialty": row[4],
                        "is_active": bool(row[5]),
                        "created_at": _dt(row[6]),
                        "updated_at": _dt(row[7]),
                    })
                
                return {"success": True, "doctors": doctors, "total": len(doctors)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_doctor_by_id(doctor_id):
    """Get a single doctor by ID."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, email, department, specialty, is_active, created_at, updated_at "
                    "FROM sme_doctors WHERE id = %s",
                    (doctor_id,)
                )
                row = cur.fetchone()
                if not row:
                    return {"success": False, "error": f"Doctor with ID {doctor_id} not found"}
                
                return {
                    "success": True,
                    "doctor": {
                        "id": row[0],
                        "name": row[1],
                        "email": row[2],
                        "department": row[3],
                        "specialty": row[4],
                        "is_active": bool(row[5]),
                        "created_at": _dt(row[6]),
                        "updated_at": _dt(row[7]),
                    }
                }
    except Exception as e:
        return {"success": False, "error": str(e)}


def add_doctor(name, department, email=None, specialty=None):
    """Add a new doctor to the system.
    
    Args:
        name: Doctor's full name (e.g., "Dr. Sarah Chen")
        department: Department name (must match DEPARTMENTS keys)
        email: Optional email address
        specialty: Optional specialty within department
    
    Returns:
        dict with success status and new doctor ID
    """
    if not name or not department:
        return {"success": False, "error": "Name and department are required"}
    
    if department not in DEPARTMENTS:
        return {"success": False, "error": f"Unknown department: {department}. Valid departments: {list(DEPARTMENTS.keys())}"}
    
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO sme_doctors (name, email, department, specialty)
                       VALUES (%s, %s, %s, %s) RETURNING id""",
                    (name, email, department, specialty)
                )
                new_id = cur.fetchone()[0]
            conn.commit()
            return {"success": True, "id": new_id, "message": f"Doctor '{name}' added successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def update_doctor(doctor_id, name=None, email=None, department=None, specialty=None, is_active=None):
    """Update an existing doctor's information.
    
    Args:
        doctor_id: ID of doctor to update
        name: New name (optional)
        email: New email (optional)
        department: New department (optional)
        specialty: New specialty (optional)
        is_active: New active status (optional)
    
    Returns:
        dict with success status
    """
    updates = []
    params = []
    
    if name is not None:
        updates.append("name = %s")
        params.append(name)
    if email is not None:
        updates.append("email = %s")
        params.append(email)
    if department is not None:
        if department not in DEPARTMENTS:
            return {"success": False, "error": f"Unknown department: {department}"}
        updates.append("department = %s")
        params.append(department)
    if specialty is not None:
        updates.append("specialty = %s")
        params.append(specialty)
    if is_active is not None:
        updates.append("is_active = %s")
        params.append(is_active if not _use_sqlite else (1 if is_active else 0))
    
    if not updates:
        return {"success": False, "error": "No fields to update"}
    
    updates.append("updated_at = %s")
    params.append(datetime.now())
    params.append(doctor_id)
    
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE sme_doctors SET {', '.join(updates)} WHERE id = %s",
                    params
                )
                if cur.rowcount == 0:
                    return {"success": False, "error": f"Doctor with ID {doctor_id} not found"}
            conn.commit()
            return {"success": True, "message": f"Doctor {doctor_id} updated successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_doctor(doctor_id, hard_delete=False):
    """Delete a doctor (soft delete by default).
    
    Args:
        doctor_id: ID of doctor to delete
        hard_delete: If True, permanently delete. If False (default), set is_active=false
    
    Returns:
        dict with success status
    """
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                if hard_delete:
                    cur.execute("DELETE FROM sme_doctors WHERE id = %s", (doctor_id,))
                    action = "deleted"
                else:
                    cur.execute(
                        "UPDATE sme_doctors SET is_active = %s, updated_at = %s WHERE id = %s",
                        (False if not _use_sqlite else 0, datetime.now(), doctor_id)
                    )
                    action = "deactivated"
                
                if cur.rowcount == 0:
                    return {"success": False, "error": f"Doctor with ID {doctor_id} not found"}
            conn.commit()
            return {"success": True, "message": f"Doctor {doctor_id} {action} successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_doctors_by_departments():
    """Get all doctors grouped by department.
    
    Returns:
        dict with departments as keys and lists of doctors as values
    """
    result = get_doctors(active_only=True)
    if not result["success"]:
        return result
    
    by_dept = {}
    for doc in result["doctors"]:
        dept = doc["department"]
        if dept not in by_dept:
            by_dept[dept] = []
        by_dept[dept].append(doc)
    
    # Ensure all departments have an entry (even if empty)
    for dept in DEPARTMENTS.keys():
        if dept not in by_dept:
            by_dept[dept] = []
    
    return {"success": True, "doctors_by_department": by_dept}


def seed_sample_doctors():
    """Seed the database with sample doctors (3 per department).
    Only adds doctors if the table is empty.
    
    Returns:
        dict with success status and count of doctors added
    """
    # Check if doctors already exist
    result = get_doctors(active_only=False)
    if result["success"] and result["total"] > 0:
        return {"success": True, "message": f"Doctors table already has {result['total']} entries, skipping seed", "added": 0}
    
    # Sample doctors: 3 per department
    sample_doctors = [
        # Cardiology
        ("Dr. Sarah Chen", "sarah.chen@hospital.org", "Cardiology", "Interventional Cardiology"),
        ("Dr. Michael Roberts", "michael.roberts@hospital.org", "Cardiology", "Electrophysiology"),
        ("Dr. Emily Johnson", "emily.johnson@hospital.org", "Cardiology", "Heart Failure"),
        # Neurology
        ("Dr. James Wilson", "james.wilson@hospital.org", "Neurology", "Stroke"),
        ("Dr. Lisa Park", "lisa.park@hospital.org", "Neurology", "Epilepsy"),
        ("Dr. David Martinez", "david.martinez@hospital.org", "Neurology", "Movement Disorders"),
        # Diabetes
        ("Dr. Amanda Thompson", "amanda.thompson@hospital.org", "Diabetes", "Type 1 Diabetes"),
        ("Dr. Robert Kim", "robert.kim@hospital.org", "Diabetes", "Type 2 Diabetes"),
        ("Dr. Jennifer Lee", "jennifer.lee@hospital.org", "Diabetes", "Diabetic Complications"),
        # Pulmonology
        ("Dr. Christopher Brown", "chris.brown@hospital.org", "Pulmonology", "COPD"),
        ("Dr. Michelle Garcia", "michelle.garcia@hospital.org", "Pulmonology", "Asthma"),
        ("Dr. Andrew Taylor", "andrew.taylor@hospital.org", "Pulmonology", "Interstitial Lung Disease"),
        # Gastroenterology
        ("Dr. Patricia White", "patricia.white@hospital.org", "Gastroenterology", "IBD"),
        ("Dr. Daniel Anderson", "daniel.anderson@hospital.org", "Gastroenterology", "Hepatology"),
        ("Dr. Rachel Moore", "rachel.moore@hospital.org", "Gastroenterology", "Endoscopy"),
        # Nephrology
        ("Dr. Steven Clark", "steven.clark@hospital.org", "Nephrology", "Dialysis"),
        ("Dr. Laura Rodriguez", "laura.rodriguez@hospital.org", "Nephrology", "Transplant"),
        ("Dr. Kevin Wright", "kevin.wright@hospital.org", "Nephrology", "Glomerular Disease"),
        # Oncology
        ("Dr. Nancy Lewis", "nancy.lewis@hospital.org", "Oncology", "Breast Cancer"),
        ("Dr. Thomas Hall", "thomas.hall@hospital.org", "Oncology", "Lung Cancer"),
        ("Dr. Karen Young", "karen.young@hospital.org", "Oncology", "Hematologic Oncology"),
        # Hematology
        ("Dr. Brian King", "brian.king@hospital.org", "Hematology", "Coagulation"),
        ("Dr. Susan Scott", "susan.scott@hospital.org", "Hematology", "Anemia"),
        ("Dr. Joseph Green", "joseph.green@hospital.org", "Hematology", "Blood Disorders"),
        # Orthopedics
        ("Dr. Elizabeth Adams", "elizabeth.adams@hospital.org", "Orthopedics", "Joint Replacement"),
        ("Dr. Richard Baker", "richard.baker@hospital.org", "Orthopedics", "Sports Medicine"),
        ("Dr. Maria Nelson", "maria.nelson@hospital.org", "Orthopedics", "Spine Surgery"),
        # Dermatology
        ("Dr. Charles Hill", "charles.hill@hospital.org", "Dermatology", "Skin Cancer"),
        ("Dr. Dorothy Carter", "dorothy.carter@hospital.org", "Dermatology", "Psoriasis"),
        ("Dr. Frank Mitchell", "frank.mitchell@hospital.org", "Dermatology", "Dermatitis"),
        # Ophthalmology
        ("Dr. Helen Perez", "helen.perez@hospital.org", "Ophthalmology", "Retina"),
        ("Dr. George Roberts", "george.roberts@hospital.org", "Ophthalmology", "Glaucoma"),
        ("Dr. Betty Turner", "betty.turner@hospital.org", "Ophthalmology", "Cataract Surgery"),
        # Psychiatry
        ("Dr. Edward Phillips", "edward.phillips@hospital.org", "Psychiatry", "Depression"),
        ("Dr. Margaret Campbell", "margaret.campbell@hospital.org", "Psychiatry", "Anxiety Disorders"),
        ("Dr. Ronald Parker", "ronald.parker@hospital.org", "Psychiatry", "Bipolar Disorder"),
        # Pediatrics
        ("Dr. Sandra Evans", "sandra.evans@hospital.org", "Pediatrics", "General Pediatrics"),
        ("Dr. Kenneth Edwards", "kenneth.edwards@hospital.org", "Pediatrics", "Neonatology"),
        ("Dr. Carol Collins", "carol.collins@hospital.org", "Pediatrics", "Pediatric Cardiology"),
        # Rheumatology
        ("Dr. Mark Stewart", "mark.stewart@hospital.org", "Rheumatology", "Rheumatoid Arthritis"),
        ("Dr. Diane Sanchez", "diane.sanchez@hospital.org", "Rheumatology", "Lupus"),
        ("Dr. Paul Morris", "paul.morris@hospital.org", "Rheumatology", "Fibromyalgia"),
        # Urology
        ("Dr. Angela Rogers", "angela.rogers@hospital.org", "Urology", "Prostate"),
        ("Dr. Timothy Reed", "timothy.reed@hospital.org", "Urology", "Kidney Stones"),
        ("Dr. Sharon Cook", "sharon.cook@hospital.org", "Urology", "Urologic Oncology"),
        # Immunology & Allergy
        ("Dr. Larry Morgan", "larry.morgan@hospital.org", "Immunology & Allergy", "Food Allergies"),
        ("Dr. Virginia Bell", "virginia.bell@hospital.org", "Immunology & Allergy", "Asthma/Allergy"),
        ("Dr. Raymond Murphy", "raymond.murphy@hospital.org", "Immunology & Allergy", "Immunodeficiency"),
        # Infectious Disease
        ("Dr. Joyce Bailey", "joyce.bailey@hospital.org", "Infectious Disease", "HIV/AIDS"),
        ("Dr. Dennis Rivera", "dennis.rivera@hospital.org", "Infectious Disease", "Hospital Infections"),
        ("Dr. Judith Cooper", "judith.cooper@hospital.org", "Infectious Disease", "Tropical Diseases"),
        # Emergency Medicine
        ("Dr. Gerald Richardson", "gerald.richardson@hospital.org", "Emergency Medicine", "Trauma"),
        ("Dr. Teresa Cox", "teresa.cox@hospital.org", "Emergency Medicine", "Critical Care"),
        ("Dr. Jerry Howard", "jerry.howard@hospital.org", "Emergency Medicine", "Toxicology"),
        # Hepatology
        ("Dr. Debra Ward", "debra.ward@hospital.org", "Hepatology", "Viral Hepatitis"),
        ("Dr. Wayne Torres", "wayne.torres@hospital.org", "Hepatology", "Liver Transplant"),
        ("Dr. Gloria Peterson", "gloria.peterson@hospital.org", "Hepatology", "Cirrhosis"),
        # Nutrition & Dietetics
        ("Dr. Roy Gray", "roy.gray@hospital.org", "Nutrition & Dietetics", "Clinical Nutrition"),
        ("Dr. Alice Ramirez", "alice.ramirez@hospital.org", "Nutrition & Dietetics", "Obesity Medicine"),
        ("Dr. Eugene James", "eugene.james@hospital.org", "Nutrition & Dietetics", "Metabolic Disorders"),
        # Pharmacology
        ("Dr. Ann Watson", "ann.watson@hospital.org", "Pharmacology", "Clinical Pharmacology"),
        ("Dr. Russell Brooks", "russell.brooks@hospital.org", "Pharmacology", "Drug Interactions"),
        ("Dr. Frances Kelly", "frances.kelly@hospital.org", "Pharmacology", "Pharmacokinetics"),
        # General Medicine
        ("Dr. Jack Sanders", "jack.sanders@hospital.org", "General Medicine", "Internal Medicine"),
        ("Dr. Ruby Price", "ruby.price@hospital.org", "General Medicine", "Primary Care"),
        ("Dr. Albert Bennett", "albert.bennett@hospital.org", "General Medicine", "Preventive Medicine"),
        # ENT (Otolaryngology)
        ("Dr. Phyllis Wood", "phyllis.wood@hospital.org", "ENT (Otolaryngology)", "Head & Neck Surgery"),
        ("Dr. Jesse Barnes", "jesse.barnes@hospital.org", "ENT (Otolaryngology)", "Hearing Disorders"),
        ("Dr. Lillian Ross", "lillian.ross@hospital.org", "ENT (Otolaryngology)", "Sinus Surgery"),
        # Obstetrics & Gynecology
        ("Dr. Howard Henderson", "howard.henderson@hospital.org", "Obstetrics & Gynecology", "High-Risk Pregnancy"),
        ("Dr. Jean Coleman", "jean.coleman@hospital.org", "Obstetrics & Gynecology", "Reproductive Medicine"),
        ("Dr. Arthur Jenkins", "arthur.jenkins@hospital.org", "Obstetrics & Gynecology", "Gynecologic Oncology"),
        # Radiology
        ("Dr. Catherine Perry", "catherine.perry@hospital.org", "Radiology", "Diagnostic Radiology"),
        ("Dr. Henry Powell", "henry.powell@hospital.org", "Radiology", "Interventional Radiology"),
        ("Dr. Ruth Long", "ruth.long@hospital.org", "Radiology", "Neuroradiology"),
        # Anesthesiology
        ("Dr. Philip Patterson", "philip.patterson@hospital.org", "Anesthesiology", "Cardiac Anesthesia"),
        ("Dr. Marie Hughes", "marie.hughes@hospital.org", "Anesthesiology", "Pediatric Anesthesia"),
        ("Dr. Ralph Flores", "ralph.flores@hospital.org", "Anesthesiology", "Pain Management"),
        # Pain Management
        ("Dr. Evelyn Washington", "evelyn.washington@hospital.org", "Pain Management", "Chronic Pain"),
        ("Dr. Louis Butler", "louis.butler@hospital.org", "Pain Management", "Interventional Pain"),
        ("Dr. Mildred Simmons", "mildred.simmons@hospital.org", "Pain Management", "Cancer Pain"),
        # Sleep Medicine
        ("Dr. Johnny Foster", "johnny.foster@hospital.org", "Sleep Medicine", "Sleep Apnea"),
        ("Dr. Doris Gonzales", "doris.gonzales@hospital.org", "Sleep Medicine", "Insomnia"),
        ("Dr. Earl Bryant", "earl.bryant@hospital.org", "Sleep Medicine", "Narcolepsy"),
        # Sports Medicine
        ("Dr. Martha Alexander", "martha.alexander@hospital.org", "Sports Medicine", "Athletic Injuries"),
        ("Dr. Bruce Russell", "bruce.russell@hospital.org", "Sports Medicine", "Rehabilitation"),
        ("Dr. Irene Griffin", "irene.griffin@hospital.org", "Sports Medicine", "Performance Medicine"),
        # Geriatrics
        ("Dr. Stanley Diaz", "stanley.diaz@hospital.org", "Geriatrics", "Dementia Care"),
        ("Dr. Hazel Hayes", "hazel.hayes@hospital.org", "Geriatrics", "Palliative Care"),
        ("Dr. Fred Myers", "fred.myers@hospital.org", "Geriatrics", "Geriatric Assessment"),
    ]
    
    added = 0
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                for name, email, dept, spec in sample_doctors:
                    cur.execute(
                        """INSERT INTO sme_doctors (name, email, department, specialty)
                           VALUES (%s, %s, %s, %s)""",
                        (name, email, dept, spec)
                    )
                    added += 1
            conn.commit()
        return {"success": True, "message": f"Seeded {added} sample doctors", "added": added}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Module initialization
# ============================================================================

if __name__ == "__main__":
    print("SFT Experiment Manager")
    print(f"External module path: {EXPERIMENTS_MODULE_PATH}")
    print(f"SFT models directory: {SFT_MODELS_DIR}")
    print(f"Module exists: {os.path.exists(EXPERIMENTS_MODULE_PATH)}")

    print("\nEnsuring database tables...")
    ensure_tables()

    print("\nChecking ranked data stats...")
    stats = get_ranked_data_stats()
    print(f"  Stats: {stats}")

    print("\nListing experiments...")
    exps = list_experiments()
    print(f"  Experiments: {exps}")
