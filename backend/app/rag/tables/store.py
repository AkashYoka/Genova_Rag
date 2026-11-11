"""Independent persistence + attachment of complete tables.

Tables are stored per user/document as complete ``TableRecord`` objects keyed by
a stable ``table_id``. Retrieval attaches the whole table to any chunk that
references it, so a hit on a caption or on one row never loses the grid.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.config import get_settings
from app.rag.tables.schema import TableRecord

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_CACHE: Dict[str, Dict[str, TableRecord]] = {}


def _store_root() -> Path:
    configured = getattr(get_settings(), "TABLE_STORE_DIR", "") or "./data/tables"
    return Path(configured)


def _document_path(user_id: str, document_id: str) -> Path:
    safe_user = str(user_id or "anonymous").replace("/", "_").replace("\\", "_")
    safe_doc = str(document_id or "document").replace("/", "_").replace("\\", "_")
    return _store_root() / safe_user / f"{safe_doc}.json"


def _cache_key(user_id: str, document_id: str) -> str:
    return f"{user_id}::{document_id}"


def save_records(
    *,
    user_id: str,
    document_id: str,
    records: Sequence[TableRecord],
) -> List[str]:
    """Persist complete tables atomically; returns stored table ids."""
    if not records:
        return []
    path = _document_path(user_id, document_id)
    payload = {
        "document_id": str(document_id),
        "tables": [record.to_json() for record in records],
    }
    with _LOCK:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Could not persist tables for document %s: %s", document_id, exc)
            return []
        _CACHE[_cache_key(user_id, document_id)] = {
            record.table_id: record for record in records
        }
    logger.info(
        "TABLE_FLOW %s",
        json.dumps(
            {
                "event": "table_store_saved",
                "document_id": document_id,
                "tables": len(records),
                "table_ids": [record.table_id for record in records],
                "path": str(path),
            }
        ),
    )
    return [record.table_id for record in records]


def load_records(*, user_id: str, document_id: str) -> List[TableRecord]:
    key = _cache_key(user_id, document_id)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return list(cached.values())
    path = _document_path(user_id, document_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read table store %s: %s", path, exc)
        return []
    records: List[TableRecord] = []
    for item in payload.get("tables") or []:
        record = TableRecord.from_json(item)
        if record is not None:
            records.append(record)
    with _LOCK:
        _CACHE[key] = {record.table_id: record for record in records}
    return records


def get_record(
    *,
    user_id: str,
    table_id: str,
    document_id: Optional[str] = None,
) -> Optional[TableRecord]:
    if not table_id:
        return None
    if document_id:
        for record in load_records(user_id=user_id, document_id=document_id):
            if record.table_id == table_id:
                return record
    root = _store_root() / str(user_id or "anonymous")
    if not root.exists():
        return None
    # Fallback search across this user's documents (table ids are globally
    # unique per table content + region).
    for path in root.glob("*.json"):
        doc_id = path.stem
        for record in load_records(user_id=user_id, document_id=doc_id):
            if record.table_id == table_id:
                return record
    return None


def delete_document_tables(*, user_id: str, document_id: str) -> None:
    with _LOCK:
        _CACHE.pop(_cache_key(user_id, document_id), None)
    path = _document_path(user_id, document_id)
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Could not delete table store %s: %s", path, exc)


def tables_from_chunks(chunks: Sequence[Dict[str, Any]], *, max_tables: int = 8) -> List[TableRecord]:
    """Collect complete tables carried by (or referenced from) retrieved chunks."""
    records: List[TableRecord] = []
    seen: set[str] = set()

    def add(record: Optional[TableRecord], *, chunk: Dict[str, Any], source_index: int) -> None:
        if record is None or not record.rows:
            return
        key = record.table_id
        if key in seen:
            return
        if not record.filename:
            record.filename = str(chunk.get("filename") or "")
        if not record.source_index:
            record.source_index = source_index
        seen.add(key)
        records.append(record)

    for index, chunk in enumerate(chunks or [], start=1):
        if not isinstance(chunk, dict):
            continue
        for key in ("complete_table", "table"):
            payload = chunk.get(key)
            if payload:
                add(TableRecord.from_json(payload), chunk=chunk, source_index=index)
        # Attached list produced by attach_complete_tables.
        for payload in chunk.get("complete_tables") or []:
            add(TableRecord.from_json(payload), chunk=chunk, source_index=index)
        if chunk.get("table_json"):
            add(TableRecord.from_json(chunk["table_json"]), chunk=chunk, source_index=index)
        if len(records) >= max_tables:
            break
    return records[:max_tables]


def attach_complete_tables(
    chunks: Sequence[Dict[str, Any]],
    *,
    user_id: str,
) -> List[Dict[str, Any]]:
    """Attach the complete table for every table-referencing chunk.

    Resolution order: the independent store (authoritative) then the chunk's own
    complete ``table_json`` (older indexes / cache misses).
    """
    if not chunks:
        return list(chunks)
    from app.rag.tables.schema import TableRecord as _TableRecord

    requested: List[str] = []
    output: List[Dict[str, Any]] = []
    attached: Dict[str, Dict[str, Any]] = {}

    for raw in chunks:
        chunk = dict(raw)
        document_id = str(chunk.get("document_id") or "")
        table_id = str(chunk.get("table_id") or "").strip()

        record: Optional[_TableRecord] = None
        if table_id and user_id:
            record = get_record(
                user_id=user_id,
                table_id=table_id,
                document_id=document_id or None,
            )
            if table_id not in requested:
                requested.append(table_id)
        if record is None:
            record = _TableRecord.from_json(chunk.get("table_json"))
        if record is None:
            payload = chunk.get("complete_table") or chunk.get("table")
            record = _TableRecord.from_json(payload)
        if record is None:
            # Page-level fallback for older chunks without ids.
            index = chunk.get("table_index")
            page = chunk.get("page")
            if index is not None and page is not None and document_id:
                for candidate in load_records(user_id=user_id, document_id=document_id):
                    if int(candidate.page) == int(page):
                        record = candidate
                        break
        if record is None:
            output.append(chunk)
            continue

        chunk["table_id"] = record.table_id
        if not chunk.get("document_id") and record.document_id:
            chunk["document_id"] = record.document_id
        chunk["complete_table"] = record.to_json()
        chunk["complete_tables"] = [record.to_json()]
        attached.setdefault(record.table_id, chunk["complete_table"])
        output.append(chunk)

    logger.info(
        "TABLE_FLOW %s",
        json.dumps(
            {
                "event": "tables_attached",
                "requested_table_ids": requested,
                "attached_table_ids": sorted(attached.keys()),
                "attached_tables": len(attached),
                "chunks": len(output),
            }
        ),
    )
    return output


def save_tables_from_chunks(
    *,
    document_id: str,
    user_id: str,
    filename: str,
    chunks: Sequence[Dict[str, Any]],
) -> List[str]:
    """Persist every complete table found in freshly chunked content."""
    records = tables_from_chunks(chunks, max_tables=1000)
    for record in records:
        record.document_id = record.document_id or str(document_id)
        if not record.filename:
            record.filename = filename
    return save_records(user_id=user_id, document_id=document_id, records=records)
