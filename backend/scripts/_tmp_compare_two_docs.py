"""Compare table extraction for two local Genova review docs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DOCS = [
    "75ec9faa-3b60-4779-9b94-ca04a6edd3f7",
    "e5fb2741-c010-48a6-a81d-fee0ac893ef9",
]
db = Path(__file__).resolve().parents[1] / "data" / "app.db"
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

for doc in DOCS:
    print("=" * 72)
    cur.execute(
        "SELECT id, filename, status FROM documents WHERE id=?",
        (doc,),
    )
    d = cur.fetchone()
    print("DOC", dict(d) if d else None)
    cur.execute(
        """
        SELECT table_id, page_start, extraction_method, index_version,
               caption, length(table_json) AS jl
        FROM document_tables WHERE document_id=?
        ORDER BY page_start, table_id
        """,
        (doc,),
    )
    rows = cur.fetchall()
    print("TABLE COUNT", len(rows))
    for r in rows:
        payload = json.loads(
            cur.execute(
                "SELECT table_json FROM document_tables WHERE document_id=? AND table_id=?",
                (doc, r["table_id"]),
            ).fetchone()["table_json"]
            or "{}"
        )
        headers = payload.get("headers") or []
        print(
            f"  {r['table_id']} method={r['extraction_method']} "
            f"ver={r['index_version']} headers={headers[:5]!r}"
        )

# Also dump relevant settings from env/config if possible via app
try:
    from app.config import get_settings

    s = get_settings()
    keys = [
        "PDF_STE_TABLES_ENABLED",
        "PDF_STE_BEDROCK_ENABLED",
        "PDF_STE_BEDROCK_MODEL_ID",
        "PDF_TEXTRACT_FALLBACK_ENABLED",
        "PDF_TEXTRACT_TABLES_ENABLED",
        "PDF_SKIP_DETECT_WHEN_STE",
        "PDF_STE_TABLES_PRIMARY",
        "PDF_TABLE_LLM_NORMALIZE_ENABLED",
    ]
    print("=" * 72)
    print("SETTINGS")
    for k in keys:
        print(f"  {k}={getattr(s, k, 'MISSING')!r}")
except Exception as exc:
    print("settings error", exc)
