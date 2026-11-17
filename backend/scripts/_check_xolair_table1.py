"""Inspect stored Table 1 for a Xolair document."""
from __future__ import annotations

import json
import sys

from app.database import SessionLocal
from sqlalchemy import text

DOC_ID = sys.argv[1] if len(sys.argv) > 1 else "43ae024f-3c0d-40f5-aa49-83e9f3682641"


def main() -> None:
    db = SessionLocal()
    try:
        names = [
            r[0]
            for r in db.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        ]
        print("db_tables:", [n for n in names if "table" in n.lower()])

        cols = db.execute(text("PRAGMA table_info(document_tables)")).fetchall()
        print("document_tables cols:", [c[1] for c in cols])

        rows = db.execute(
            text(
                """
                SELECT id, page_start, page_end, caption, headers_json,
                       rows_json, n_rows, n_cols
                FROM document_tables
                WHERE document_id = :d
                ORDER BY page_start, id
                LIMIT 8
                """
            ),
            {"d": DOC_ID},
        ).fetchall()
        print("stored_tables:", len(rows))
        for row in rows:
            caption = row[3] or ""
            headers = json.loads(row[4] or "[]")
            body = json.loads(row[5] or "[]")
            print("---")
            print("id", row[0], "pages", row[1], row[2], "shape", row[6], "x", row[7])
            print("caption:", caption[:120])
            print("headers:", headers)
            if "Table 1" in caption or (headers and "300-400" in json.dumps(body)):
                for i, r in enumerate(body):
                    print(f"R{i}", r)
    finally:
        db.close()


if __name__ == "__main__":
    main()
