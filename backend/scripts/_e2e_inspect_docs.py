"""Inspect indexed documents for AI usage e2e testing."""
from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "app.db"


def main() -> None:
    print(f"db={DB} exists={DB.exists()}")
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    tables = {
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        )
    }
    print("tables=", sorted(tables))
    print("\nUSERS:")
    for r in con.execute(
        "SELECT id, username, email, is_admin, role FROM users"
    ):
        print(dict(r))
    print("\nDOCS:")
    sql = """
        SELECT id, user_id, original_name, status, page_count, chunk_count,
               substr(coalesce(summary, ''), 1, 280) AS summary,
               substr(coalesce(keywords, ''), 1, 160) AS keywords
        FROM documents
        WHERE coalesce(is_deleted, 0) = 0
        ORDER BY uploaded_at DESC
        LIMIT 80
    """
    for r in con.execute(sql):
        print(dict(r))
    if "llm_usage" in tables:
        print(
            "\nLLM_USAGE count=",
            con.execute("SELECT COUNT(*) FROM llm_usage").fetchone()[0],
        )
    else:
        print("\nLLM_USAGE table missing")
    con.close()


if __name__ == "__main__":
    main()
