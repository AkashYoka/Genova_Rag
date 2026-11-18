"""Find table chunk markdown for pediatric Table 2 frequency cells."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DOC = "f3f79e33-2650-4292-bccd-44138e65d6f8"
db = Path(__file__).resolve().parents[1] / "data" / "app.db"
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
print("tables:", [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")])

# Try common chunk stores
for table in ("document_chunks", "chunks", "hierarchy_chunks", "document_sections"):
    try:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        print(table, cols[:20])
    except Exception as e:
        print(table, "missing", e)

# Chroma is separate; check document_sections for table markdown
try:
    rows = conn.execute(
        """
        SELECT section_id, title, page_start, page_end
        FROM document_sections WHERE document_id=?
        ORDER BY page_start
        """,
        (DOC,),
    ).fetchall()
    print("sections", len(rows))
    for r in rows:
        print(dict(r))
except Exception as e:
    print("sections err", e)

# hierarchy chunk content
for name in ("document_hierarchy_chunks", "section_chunks", "chunk_hierarchy"):
    try:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({name})")]
        print("found", name, cols)
    except Exception:
        pass
