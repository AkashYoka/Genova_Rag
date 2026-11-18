"""Inspect Xolair local tables for missing Every 2 weeks cells."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DOC = "b93f5647-7bb4-4c21-8d6d-b759ef77b951"
db = Path(__file__).resolve().parents[1] / "data" / "app.db"
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute(
    """
    SELECT table_id, page_start, caption, extraction_method, length(table_json) jl
    FROM document_tables WHERE document_id=?
    ORDER BY page_start, table_id
    """,
    (DOC,),
)
rows = cur.fetchall()
print("TABLE COUNT", len(rows))
for r in rows:
    print("=" * 72)
    print(dict(r))
    payload = json.loads(
        cur.execute(
            "SELECT table_json FROM document_tables WHERE document_id=? AND table_id=?",
            (DOC, r["table_id"]),
        ).fetchone()["table_json"]
        or "{}"
    )
    headers = payload.get("headers") or []
    body = payload.get("rows") or []
    print("headers", headers[:8])
    # Find freq-like column
    freq_cols = [
        i
        for i, h in enumerate(headers)
        if "freq" in str(h).lower()
        or "every" in str(h).lower()
        or "week" in str(h).lower()
        or "dosing" in str(h).lower()
    ]
    print("freq_cols", freq_cols)
    for i, row in enumerate(body):
        if any(
            ">900" in str(c)
            or ">1000" in str(c)
            or ">1100" in str(c)
            or ">1200" in str(c)
            or "900-1000" in str(c)
            or "1100" in str(c)
            or "1200" in str(c)
            for c in row[:2]
        ):
            print(f"  row{i}", row[:6])
    # Also print last 8 rows first 4 cols
    print("--- last/high IgE rows ---")
    for i, row in enumerate(body):
        label = str(row[0] if row else "")
        if any(x in label for x in ("900", "1000", "1100", "1200", "1300")):
            print(f"  [{i}] {row[:5]}")
    prov = payload.get("provenance") or {}
    print(
        "bedrock",
        (prov.get("bedrock") or {}),
        "warnings",
        (payload.get("warnings") or prov.get("warnings") or [])[:8],
    )
