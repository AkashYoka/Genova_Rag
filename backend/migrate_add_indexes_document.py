"""
One-time migration script to add indexes on documents.user_id and documents.filename.
Run this from the 'backend' directory.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import engine
from sqlalchemy import inspect, text

INDEXES = [
    ("ix_documents_user_id", "user_id"),
    ("ix_documents_filename", "filename"),
]

def migrate():
    print("🚀 Starting migration: adding indexes on 'documents' table...")
    try:
        inspector = inspect(engine)
        existing = {index["name"] for index in inspector.get_indexes("documents")}

        with engine.begin() as conn:
            for index_name, column in INDEXES:
                if index_name in existing:
                    print(f"ℹ️  Index '{index_name}' already exists — skipping.")
                    continue
                # `CREATE INDEX IF NOT EXISTS` is PostgreSQL/SQLite only;
                # MySQL has no such clause, so existence is checked above.
                conn.execute(text(
                    f"CREATE INDEX {index_name} ON documents ({column})"
                ))
        print("✅ Migration successful!")
    except Exception as e:
        print(f"❌ Migration failed: {e}")

if __name__ == "__main__":
    migrate()