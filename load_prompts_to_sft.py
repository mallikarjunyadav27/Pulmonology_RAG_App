#!/usr/bin/env python3
"""
Load neurologist_prompts_100.json and cardiologist_prompts_100.json into
sft_ranked_data in PostgreSQL (pces_base).

Each prompt is inserted as a rank-1 placeholder row with a blank response_text.
Responses should be generated and updated afterwards.
Duplicate prompts (matched by exact text) are skipped.
"""

import json
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg as _pg
    _PG_VERSION = 3
except ImportError:
    try:
        import psycopg2 as _pg  # type: ignore
        _PG_VERSION = 2
    except ImportError:
        print("ERROR: neither psycopg nor psycopg2 is installed.")
        sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_raw = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
# psycopg uses 'dbname'; psycopg2 accepts both but 'database' is canonical
_raw["dbname" if _PG_VERSION == 3 else "database"] = os.getenv("DB_NAME")

DB_CONFIG = {k: v for k, v in _raw.items() if v is not None}

FILES = [
    ("neurologist_prompts_100.json", "Neurology"),
    ("cardiologist_prompts_100.json", "Cardiology"),
    ("pediatrician_prompts_100.json", "Pediatrics"),
    ("orthopedic_prompts_100.json", "Orthopedics"),
]


def ensure_domain_column(cur):
    """Add domain column if it doesn't exist yet (idempotent)."""
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'sft_ranked_data' AND column_name = 'domain'"
    )
    if not cur.fetchone():
        cur.execute("ALTER TABLE sft_ranked_data ADD COLUMN domain TEXT")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_domain "
            "ON sft_ranked_data(domain)"
        )
        print("  ℹ️  Added missing 'domain' column to sft_ranked_data")


def insert_prompts(cur, prompts, domain):
    inserted = 0
    skipped = 0
    for item in prompts:
        prompt_text = item.get("prompt", "").strip()
        if not prompt_text:
            continue

        cur.execute(
            "SELECT 1 FROM sft_ranked_data WHERE prompt = %s LIMIT 1",
            (prompt_text,),
        )
        if cur.fetchone():
            skipped += 1
            continue

        group_id = str(uuid.uuid4())[:8]
        cur.execute(
            """INSERT INTO sft_ranked_data
               (prompt, response_text, rank, reason, group_id, domain)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (prompt_text, "", 1, "", group_id, domain),
        )
        inserted += 1

    return inserted, skipped


def main():
    host = DB_CONFIG.get("host")
    dbname = DB_CONFIG.get("dbname") or DB_CONFIG.get("database")
    port = DB_CONFIG.get("port")
    print(f"Connecting to PostgreSQL at {host}:{port}/{dbname} ...")
    try:
        conn = _pg.connect(**DB_CONFIG)
    except Exception as e:
        print(f"ERROR: Could not connect — {e}")
        sys.exit(1)

    total_inserted = 0
    total_skipped = 0

    try:
        with conn.cursor() as cur:
            ensure_domain_column(cur)

        for filename, domain in FILES:
            path = os.path.join(SCRIPT_DIR, filename)
            print(f"\n📂  {filename}  →  domain: {domain}")
            try:
                with open(path) as f:
                    prompts = json.load(f)
            except FileNotFoundError:
                print(f"  ERROR: file not found at {path}")
                continue

            print(f"  Found {len(prompts)} prompts")
            with conn.cursor() as cur:
                inserted, skipped = insert_prompts(cur, prompts, domain)
            conn.commit()
            print(f"  ✅ Inserted: {inserted}   ⏭  Skipped (duplicate): {skipped}")
            total_inserted += inserted
            total_skipped += skipped

    finally:
        conn.close()

    print(f"\n{'='*50}")
    print(f"Total inserted : {total_inserted}")
    print(f"Total skipped  : {total_skipped}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
