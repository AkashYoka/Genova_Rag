"""Inspect new Xolair doc frequency column after reindex."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DOC = "f3f79e33-2650-4292-bccd-44138e65d6f8"
db = Path(__file__).resolve().parents[1] / "data" / "app.db"
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute(
    "SELECT id, filename, status FROM documents WHERE id=?",
    (DOC,),
)
print("DOC", dict(cur.fetchone() or {}))
cur.execute(
    """
    SELECT table_id, page_start, caption, extraction_method
    FROM document_tables WHERE document_id=?
    ORDER BY page_start, table_id
    """,
    (DOC,),
)
for r in cur.fetchall():
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
    print("headers[0:3]", headers[:3])
    print("freq column:")
    for i, row in enumerate(body):
        print(f"  [{i:02d}] {row[0]!r:20} | {row[1]!r}")
    prov = payload.get("provenance") or {}
    print("bedrock", prov.get("bedrock"))
    warns = payload.get("warnings") or prov.get("warnings") or []
    print("warnings", [w for w in warns if "shared" in str(w) or "bedrock" in str(w)][:12])
