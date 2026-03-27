#!/usr/bin/env python3
"""
Generate LLM responses for all sft_ranked_data rows that have an empty
response_text (inserted by load_prompts_to_sft.py).

For each prompt this script:
  1. Calls GPT-4o-mini to produce 3 quality-ranked answers (HIGH / MEDIUM / LOW)
  2. Updates the existing rank-1 row with the HIGH-quality answer
  3. Inserts rank-2 (MEDIUM) and rank-3 (LOW) rows under the same group_id

Domain-specific system prompts are used so the model stays in the right
specialist context (Neurology / Cardiology).
"""

import os
import re
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# DB driver — prefer psycopg (v3), fall back to psycopg2
# ---------------------------------------------------------------------------
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

_raw = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
_raw["dbname" if _PG_VERSION == 3 else "database"] = os.getenv("DB_NAME")
DB_CONFIG = {k: v for k, v in _raw.items() if v is not None}

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

client = OpenAI(
    api_key=os.getenv("openai_api_key"),
    base_url=os.getenv("base_url", "https://api.openai.com/v1"),
)
MODEL = os.getenv("llm_model_name", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Domain-specific system prompts
# ---------------------------------------------------------------------------
DOMAIN_SYSTEM_PROMPTS = {
    "Neurology": (
        "You are a board-certified neurologist providing evidence-based clinical "
        "guidance. Your responses address diagnosis, investigation, and management "
        "of neurological conditions. Always write in plain clinical prose — no "
        "code, markdown headers, or bullet points beyond brief lists where essential."
    ),
    "Cardiology": (
        "You are a board-certified cardiologist providing evidence-based clinical "
        "guidance. Your responses address diagnosis, investigation, and management "
        "of cardiovascular conditions. Always write in plain clinical prose — no "
        "code, markdown headers, or bullet points beyond brief lists where essential."
    ),
    "Pediatrics": (
        "You are a board-certified pediatrician providing evidence-based clinical "
        "guidance for patients from newborns through adolescents. Your responses "
        "address age-appropriate diagnosis, investigation, and management of "
        "pediatric conditions, including dosing adjustments and developmental "
        "considerations. Always write in plain clinical prose — no code, markdown "
        "headers, or bullet points beyond brief lists where essential."
    ),
    "Orthopedics": (
        "You are a board-certified orthopedic surgeon providing evidence-based "
        "clinical guidance. Your responses address diagnosis, imaging, conservative "
        "and surgical management of musculoskeletal conditions including fractures, "
        "joint disorders, spine pathology, and sports injuries. Always write in "
        "plain clinical prose — no code, markdown headers, or bullet points beyond "
        "brief lists where essential."
    ),
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a board-certified physician providing evidence-based clinical guidance. "
    "Write in plain clinical prose."
)

RANK_REASONS = {
    1: "Comprehensive, evidence-based response with specific details and clinical guidelines",
    2: "Adequate response covering key points with moderate detail",
    3: "Brief response — lacks specific details and clinical depth",
}

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

GENERATION_PROMPT_TMPL = """Question: {prompt}

Generate exactly 3 clinical answers of varying quality. Separate each answer with "|||".

Answer 1 (HIGH quality — 5/5):
Write a detailed, comprehensive answer: specific condition workup, medications with mechanisms, diagnostic criteria, and management guidelines.

Answer 2 (MEDIUM quality — 3/5):
Write a good but less detailed answer covering the main clinical points.

Answer 3 (LOW quality — 1/5):
Write a very brief, vague answer with minimal useful clinical information.

Format your reply as:
[Answer 1 text]|||[Answer 2 text]|||[Answer 3 text]"""


def generate_responses(prompt: str, domain: str) -> list[dict]:
    """Call the LLM and return a list of 3 dicts: {rank, text, reason}."""
    system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, DEFAULT_SYSTEM_PROMPT)
    user_msg = GENERATION_PROMPT_TMPL.format(prompt=prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        timeout=60,
    )
    raw = response.choices[0].message.content.strip()

    # --- parse "|||" delimited answers ---
    parts = [p.strip() for p in raw.split("|||") if p.strip()]

    # Fallback: split by "Answer N" markers if ||| was not used
    if len(parts) < 3:
        matches = list(re.finditer(r'Answer\s+\d+[:\s\(]', raw, re.IGNORECASE))
        if len(matches) >= 2:
            extracted = []
            for i, m in enumerate(matches):
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
                text = raw[start:end].strip()
                text = re.sub(r'^\[.*?\]\s*', '', text)
                if text:
                    extracted.append(text)
            if len(extracted) >= 2:
                parts = extracted

    # Pad to 3 if we still don't have enough
    while len(parts) < 3:
        parts.append(parts[-1] if parts else "No response generated.")

    return [
        {"rank": i + 1, "text": parts[i], "reason": RANK_REASONS[i + 1]}
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dbname = DB_CONFIG.get("dbname") or DB_CONFIG.get("database")
    print(f"Connecting to {DB_CONFIG.get('host')}:{DB_CONFIG.get('port')}/{dbname} ...")
    try:
        conn = _pg.connect(**DB_CONFIG)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Fetch all rows with empty response_text
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, prompt, group_id, domain "
            "FROM sft_ranked_data "
            "WHERE response_text = '' "
            "ORDER BY domain, id"
        )
        rows = cur.fetchall()

    if not rows:
        print("No rows with empty response_text found — nothing to do.")
        conn.close()
        return

    total = len(rows)
    print(f"Found {total} prompts needing responses.\n")

    updated = 0
    errors = 0

    for idx, (row_id, prompt, group_id, domain) in enumerate(rows, 1):
        domain = domain or "General"
        print(f"[{idx:>3}/{total}] {domain[:12]:<12} | group {group_id} …", end=" ", flush=True)

        try:
            answers = generate_responses(prompt, domain)
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
            time.sleep(2)
            continue

        try:
            with conn.cursor() as cur:
                # Update the existing rank-1 placeholder row
                cur.execute(
                    "UPDATE sft_ranked_data "
                    "SET response_text = %s, reason = %s, updated_at = NOW() "
                    "WHERE id = %s",
                    (answers[0]["text"], answers[0]["reason"], row_id),
                )

                # Insert rank-2 and rank-3 rows under the same group_id
                for ans in answers[1:]:
                    cur.execute(
                        "INSERT INTO sft_ranked_data "
                        "(prompt, response_text, rank, reason, group_id, domain) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (prompt, ans["text"], ans["rank"], ans["reason"], group_id, domain),
                    )
            conn.commit()
            updated += 1
            print("✅")
        except Exception as e:
            conn.rollback()
            print(f"DB ERROR: {e}")
            errors += 1

        # Polite pacing — avoid OpenAI rate limits
        time.sleep(0.5)

    conn.close()

    print(f"\n{'='*55}")
    print(f"Prompts processed : {updated}")
    print(f"Rows inserted      : {updated * 2}  (rank-2 + rank-3 per prompt)")
    print(f"Errors             : {errors}")
    print(f"Total rows now     : {total * 3 - errors * 2} (approx)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
