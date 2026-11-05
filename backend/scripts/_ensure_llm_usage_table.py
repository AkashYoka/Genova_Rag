"""Ensure llm_usage table exists for the live SQLite DB."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.database import Base, engine
from app import models  # noqa: F401 — register metadata


def main() -> None:
    Base.metadata.create_all(bind=engine, tables=[models.LLMUsage.__table__])
    db = Path(__file__).resolve().parents[1] / "data" / "app.db"
    con = sqlite3.connect(db)
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    print("has llm_usage=", "llm_usage" in names)
    if "llm_usage" in names:
        print("columns=", con.execute("PRAGMA table_info(llm_usage)").fetchall())
    con.close()


if __name__ == "__main__":
    main()
