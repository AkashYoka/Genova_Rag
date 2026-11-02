"""migrate_postgres_to_mysql.py
--------------------------------
Copy every row from an existing PostgreSQL deployment into MySQL 8.4 LTS
without data loss, then validate the result.

The destination schema is created from the application's SQLAlchemy models
(``app.models``), so the copy is always done against the schema the app
itself expects. Rows are inserted in foreign-key-safe order, UUIDs are
normalised to their canonical string form, and timezone-aware timestamps are
converted to naive UTC (MySQL ``DATETIME`` has no timezone).

Dependencies
------------
    pip install "pymysql" "psycopg[binary]"   # driver is migration-only
    # psycopg is NOT part of backend/requirements.txt any more — the
    # application itself only needs PyMySQL.

Safety model
------------
* ``--backup PATH`` writes a full JSON snapshot of the PostgreSQL source
  *before* anything is written to MySQL.
* The copy runs inside a single MySQL transaction; any error rolls the whole
  copy back, and re-running is safe (existing primary keys are skipped).
* ``--dry-run`` touches nothing: it only reports per-table row counts.
* After the copy, per-table row counts are compared (hard failure) and a
  sample of rows is compared column by column (reported per table).

Usage
-----
    # 1. Snapshot the old database first
    docker exec pdf_rag_postgres pg_dump -U pdf_rag_user -d pdf_rag -Fc \
        > backup_pre_mysql_$(date +%Y%m%d).dump

    # 2. Dry run (no writes)
    python backend/scripts/migrate_postgres_to_mysql.py \
        --postgres postgresql://pdf_rag_user:pdf_rag_pass@localhost:5432/pdf_rag \
        --mysql    mysql+pymysql://pdf_rag_user:pdf_rag_pass@localhost:3306/pdf_rag \
        --dry-run

    # 3. Live migration with a JSON snapshot
    python backend/scripts/migrate_postgres_to_mysql.py \
        --postgres postgresql://pdf_rag_user:pdf_rag_pass@localhost:5432/pdf_rag \
        --mysql    mysql+pymysql://pdf_rag_user:pdf_rag_pass@localhost:3306/pdf_rag \
        --backup ./backups/pre_mysql.json

    # 4. Re-check a completed migration without writing
    python backend/scripts/migrate_postgres_to_mysql.py ... --verify-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.exc import SQLAlchemyError

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pg_to_mysql")


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def normalize_postgres_url(url: str) -> str:
    """Add the psycopg (v3) driver specifier to a plain PostgreSQL URL."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def normalize_mysql_url(url: str) -> str:
    """Add the PyMySQL driver specifier to a plain MySQL URL."""
    for prefix in ("mysql://", "mariadb://"):
        if url.startswith(prefix):
            return "mysql+pymysql://" + url[len(prefix):]
    return url


def _make_engine(url: str, label: str):
    kwargs: Dict[str, Any] = {"echo": False, "future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(pool_size=5, max_overflow=10, pool_recycle=3600)
        if url.startswith("mysql"):
            kwargs["connect_args"] = {"charset": "utf8mb4"}
    logger.info("Connecting to %s: %s", label, url)
    return create_engine(url, **kwargs)


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------

def _coerce_value(value: Any) -> Any:
    """Convert a PostgreSQL value into something MySQL accepts verbatim."""
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, memoryview):
        return bytes(value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return value


def _compare_key(value: Any) -> Any:
    """Normalise a value so PostgreSQL and MySQL representations compare equal."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, float):
        return round(value, 9)
    return value


def _normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _coerce_value(value) for key, value in row.items()}


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _load_metadata() -> MetaData:
    """Import the ORM models and return their metadata."""
    from app import models  # noqa: F401 — registers every table on Base.metadata
    from app.database import Base

    return Base.metadata


def _source_tables(metadata: MetaData, src_inspector) -> Tuple[List[Table], List[str]]:
    ordered: List[Table] = []
    missing: List[str] = []
    for table in metadata.sorted_tables:  # FK-safe order
        if src_inspector.has_table(table.name):
            ordered.append(table)
        else:
            missing.append(table.name)
    return ordered, missing


def _row_count(conn, table: Table) -> int:
    return int(conn.execute(text(f"SELECT COUNT(*) FROM {table.name}")).scalar() or 0)


def _truncate_destination(dst_engine, tables: Sequence[Table]) -> None:
    with dst_engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for table in reversed(list(tables)):
            conn.execute(text(f"TRUNCATE TABLE `{table.name}`"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
    logger.info("Truncated %d destination tables", len(tables))


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def _write_backup(path: Path, tables: Sequence[Table], src_engine, source_url: str) -> None:
    payload: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source_url,
        "tables": {},
    }
    with src_engine.connect() as conn:
        for table in tables:
            pk_cols = list(table.primary_key.columns)
            stmt = select(table)
            if pk_cols:
                stmt = stmt.order_by(*pk_cols)
            rows = [
                {key: _jsonable(value) for key, value in row.items()}
                for row in conn.execute(stmt).mappings()
            ]
            payload["tables"][table.name] = rows
            logger.info("  backup: %-28s %d rows", table.name, len(rows))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Source snapshot written to %s", path)


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def _copy_table(
    table: Table,
    src_conn,
    dst_conn,
    batch_size: int,
) -> Dict[str, int]:
    """Copy one table, skipping rows whose primary key already exists."""
    stats = {"read": 0, "inserted": 0, "skipped": 0}
    pk_cols = list(table.primary_key.columns)

    stmt = select(table)
    if pk_cols:
        stmt = stmt.order_by(*pk_cols)

    existing: set = set()
    if pk_cols:
        existing = {
            tuple(_compare_key(value) for value in row)
            for row in dst_conn.execute(select(*pk_cols)).all()
        }

    insert_stmt = table.insert()
    batch: List[Dict[str, Any]] = []

    for row in src_conn.execute(stmt).mappings():
        stats["read"] += 1
        row_dict = _normalise_row(dict(row))

        if pk_cols:
            key = tuple(_compare_key(row_dict[col.name]) for col in pk_cols)
            if key in existing:
                stats["skipped"] += 1
                continue
            existing.add(key)

        batch.append(row_dict)
        if len(batch) >= batch_size:
            dst_conn.execute(insert_stmt, batch)
            stats["inserted"] += len(batch)
            batch = []

    if batch:
        dst_conn.execute(insert_stmt, batch)
        stats["inserted"] += len(batch)

    return stats


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_table(
    table: Table,
    src_conn,
    dst_conn,
    sample: int,
) -> Tuple[bool, List[str]]:
    problems: List[str] = []

    src_count = _row_count(src_conn, table)
    dst_count = _row_count(dst_conn, table)
    if src_count != dst_count:
        problems.append(f"row count mismatch: source={src_count} destination={dst_count}")

    pk_cols = list(table.primary_key.columns)
    if pk_cols and sample > 0:
        src_rows = src_conn.execute(
            select(table).order_by(*pk_cols).limit(sample)
        ).mappings().all()
        for src_row in src_rows:
            filters = [col == src_row[col.name] for col in pk_cols]
            dst_row = dst_conn.execute(select(table).where(*filters)).mappings().first()
            if dst_row is None:
                problems.append(f"missing row with pk {[src_row[c.name] for c in pk_cols]}")
                continue
            for column in table.columns:
                if _compare_key(src_row[column.name]) != _compare_key(dst_row[column.name]):
                    problems.append(
                        f"column {column.name!r} differs for pk "
                        f"{[src_row[c.name] for c in pk_cols]}"
                    )
                    break

    return (not problems), problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate all data from PostgreSQL to MySQL 8.4 LTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--postgres",
        metavar="URL",
        default=(
            os.getenv("POSTGRES_DATABASE_URL")
            or os.getenv("POSTGRES_URL")
            or "postgresql://pdf_rag_user:pdf_rag_pass@localhost:5432/pdf_rag"
        ),
        help="PostgreSQL source URL (postgresql:// or postgres://).",
    )
    parser.add_argument(
        "--mysql",
        metavar="URL",
        default=(
            os.getenv("MYSQL_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or "mysql+pymysql://pdf_rag_user:pdf_rag_pass@localhost:3306/pdf_rag"
        ),
        help="MySQL destination URL (mysql:// is upgraded to mysql+pymysql://).",
    )
    parser.add_argument(
        "--backup",
        metavar="PATH",
        default=None,
        help="Write a JSON snapshot of the PostgreSQL source to PATH before copying.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="TRUNCATE destination tables before copying (destructive — back up first).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report source row counts; write nothing.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only compare source and destination; write nothing.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Rows per table compared column by column during verification (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows per INSERT batch (default: 500).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    source_url = normalize_postgres_url(args.postgres)
    destination_url = normalize_mysql_url(args.mysql)

    logger.info("=" * 68)
    logger.info("PostgreSQL → MySQL 8.4 data migration")
    logger.info("Source      : %s", source_url)
    logger.info("Destination : %s", destination_url)
    logger.info("Mode        : %s", "DRY RUN" if args.dry_run else (
        "VERIFY ONLY" if args.verify_only else "LIVE"))
    logger.info("=" * 68)

    src_engine = _make_engine(source_url, "PostgreSQL")
    dst_engine = _make_engine(destination_url, "MySQL")

    try:
        with src_engine.connect() as probe:
            probe.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.error("Cannot connect to PostgreSQL: %s", exc)
        return 1

    metadata = _load_metadata()
    src_inspector = inspect(src_engine)
    tables, missing = _source_tables(metadata, src_inspector)
    if missing:
        logger.info("Tables absent in source (treated as empty): %s", ", ".join(missing))
    if not tables:
        logger.error("No known tables found in the PostgreSQL source — aborting.")
        return 1

    if args.dry_run:
        with src_engine.connect() as src_conn:
            total = 0
            for table in tables:
                count = _row_count(src_conn, table)
                total += count
                logger.info("  %-28s %d rows", table.name, count)
            logger.info("-" * 68)
            logger.info("  TOTAL %d rows — nothing was written", total)
        return 0

    if args.verify_only:
        return 0 if _run_verification(tables, src_engine, dst_engine, args.sample) else 1

    try:
        with dst_engine.connect() as probe:
            probe.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.error("Cannot connect to MySQL: %s", exc)
        return 1

    if args.backup:
        _write_backup(Path(args.backup), tables, src_engine, source_url)
    else:
        logger.warning(
            "No --backup path given. Take a pg_dump/mysqldump snapshot before a "
            "production migration."
        )

    # Create any missing destination tables/indexes from the ORM models.
    metadata.create_all(bind=dst_engine)
    logger.info("Destination schema present (%d tables in ORM metadata)", len(metadata.tables))

    if args.truncate:
        _truncate_destination(dst_engine, tables)

    total_inserted = 0
    total_skipped = 0
    try:
        # One transaction for the whole copy: any failure rolls everything back.
        with src_engine.connect() as src_conn, dst_engine.begin() as dst_conn:
            for table in tables:
                stats = _copy_table(table, src_conn, dst_conn, args.batch_size)
                total_inserted += stats["inserted"]
                total_skipped += stats["skipped"]
                logger.info(
                    "  %-28s read=%-6d inserted=%-6d skipped=%d",
                    table.name,
                    stats["read"],
                    stats["inserted"],
                    stats["skipped"],
                )
    except Exception as exc:  # noqa: BLE001 — surface any failure and stop
        logger.error("Migration failed — MySQL transaction rolled back: %s", exc, exc_info=True)
        return 1

    logger.info("-" * 68)
    logger.info("  TOTAL inserted=%d skipped=%d", total_inserted, total_skipped)

    if not _run_verification(tables, src_engine, dst_engine, args.sample):
        logger.error("Verification FAILED — PostgreSQL remains the source of truth.")
        return 1

    logger.info("Migration verified. MySQL is ready to be promoted.")
    return 0


def _run_verification(tables, src_engine, dst_engine, sample: int) -> bool:
    logger.info("Verifying migration …")
    ok = True
    with src_engine.connect() as src_conn, dst_engine.connect() as dst_conn:
        for table in tables:
            table_ok, problems = _verify_table(table, src_conn, dst_conn, sample)
            if table_ok:
                logger.info("  %-28s OK", table.name)
            else:
                ok = False
                for problem in problems[:5]:
                    logger.error("  %-28s %s", table.name, problem)
    return ok


if __name__ == "__main__":
    sys.exit(main())
