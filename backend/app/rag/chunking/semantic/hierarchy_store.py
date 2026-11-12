"""Persistence helpers for the relational section/chunk navigation index."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from app.models import DocumentChunk, DocumentSection

_METADATA_KEYS = (
    "page_start",
    "page_end",
    "chunk_type",
    "content_type",
    "section",
    "subsection",
    "extraction_method",
    "section_path",
    "section_level",
    "parent_section_id",
    "source_block_ids",
    "table_id",
    "table_index",
    "table_part",
)


def _json(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def delete_hierarchy(document_id: str, db=None) -> None:
    """Remove all navigation rows for a document (chunks first, then sections)."""
    owns = db is None
    if owns:
        from app.database import SessionLocal

        db = SessionLocal()
    try:
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).delete(synchronize_session=False)
        db.query(DocumentSection).filter(
            DocumentSection.document_id == document_id
        ).delete(synchronize_session=False)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        if owns:
            db.close()


def persist_hierarchy(
    *,
    document_id: str,
    user_id: str,
    sections: Iterable[Any],
    chunks: Iterable[Dict[str, Any]],
    index_version: str = "",
    db=None,
) -> Dict[str, int]:
    """Replace the document's section tree and persisted chunk rows."""
    owns = db is None
    if owns:
        from app.database import SessionLocal

        db = SessionLocal()

    try:
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).delete(synchronize_session=False)
        db.query(DocumentSection).filter(
            DocumentSection.document_id == document_id
        ).delete(synchronize_session=False)
        db.flush()

        section_rows: List[DocumentSection] = []
        section_ids: set[str] = set()
        for node in sections:
            section_rows.append(
                DocumentSection(
                    id=node.section_id,
                    document_id=document_id,
                    user_id=user_id,
                    parent_section_id=node.parent_section_id,
                    title=node.title,
                    level=int(node.level or 1),
                    ordinal=int(node.ordinal or 0),
                    page_start=node.page_start,
                    page_end=node.page_end,
                    summary=node.summary or None,
                    block_ids_json=_json(list(node.block_ids)),
                    tables_json=_json(list(node.tables)),
                    chunk_count=0,
                    index_version=index_version or None,
                )
            )
            section_ids.add(node.section_id)
        if section_rows:
            db.add_all(section_rows)
            db.flush()

        counts: Dict[str, int] = {}
        chunk_rows: List[DocumentChunk] = []
        for chunk in chunks:
            section_id = chunk.get("section_id")
            if section_id not in section_ids:
                section_id = None
            chunk_index = int(chunk.get("chunk_index") or 0)
            chunk_id = str(chunk.get("id") or f"{document_id}_{chunk_index}")
            metadata = {
                key: chunk.get(key)
                for key in _METADATA_KEYS
                if chunk.get(key) not in (None, "")
            }
            table_ids = chunk.get("table_ids")
            if isinstance(table_ids, str):
                try:
                    table_ids = json.loads(table_ids)
                except Exception:
                    table_ids = [table_ids]
            chunk_rows.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    user_id=user_id,
                    section_id=section_id,
                    parent_section_id=chunk.get("parent_section_id"),
                    chunk_index=chunk_index,
                    content=chunk.get("text") or "",
                    chunk_type=str(chunk.get("chunk_type") or "text"),
                    content_type=chunk.get("content_type"),
                    page_start=chunk.get("page_start") or chunk.get("page"),
                    page_end=chunk.get("page_end") or chunk.get("page"),
                    token_count=int(chunk.get("token_count") or 0) or None,
                    block_ids_json=_json(
                        _load_json(chunk.get("source_block_ids")) or chunk.get("source_block_ids")
                    ),
                    table_ids_json=_json(table_ids) if table_ids else None,
                    metadata_json=_json(metadata),
                    index_version=index_version or None,
                )
            )
            if section_id:
                counts[section_id] = counts.get(section_id, 0) + 1

        if chunk_rows:
            db.add_all(chunk_rows)
        for row in section_rows:
            row.chunk_count = counts.get(row.id, 0)
        db.commit()
        return {"sections": len(section_rows), "chunks": len(chunk_rows)}
    except Exception:
        db.rollback()
        raise
    finally:
        if owns:
            db.close()


def _load_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def load_sections(document_id: str, db) -> List[DocumentSection]:
    return (
        db.query(DocumentSection)
        .filter(DocumentSection.document_id == document_id)
        .order_by(DocumentSection.ordinal.asc())
        .all()
    )


def load_section(document_id: str, section_id: str, db) -> Optional[DocumentSection]:
    return (
        db.query(DocumentSection)
        .filter(
            DocumentSection.document_id == document_id,
            DocumentSection.id == section_id,
        )
        .first()
    )


def load_section_chunks(
    document_id: str,
    section_id: Optional[str],
    db,
    *,
    limit: int = 500,
) -> List[DocumentChunk]:
    query = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id)
    if section_id is not None:
        query = query.filter(DocumentChunk.section_id == section_id)
    return query.order_by(DocumentChunk.chunk_index.asc()).limit(limit).all()


def serialize_section(row: DocumentSection, *, children: Optional[List[Any]] = None) -> Dict[str, Any]:
    return {
        "section_id": row.id,
        "document_id": str(row.document_id),
        "parent_section_id": row.parent_section_id,
        "title": row.title,
        "level": row.level,
        "ordinal": row.ordinal,
        "page_start": row.page_start,
        "page_end": row.page_end,
        "summary": row.summary or "",
        "block_ids": _load_json(row.block_ids_json) or [],
        "tables": _load_json(row.tables_json) or [],
        "chunk_count": row.chunk_count or 0,
        "index_version": row.index_version,
        "children": children or [],
    }


def serialize_chunk(row: DocumentChunk, *, include_content: bool = True) -> Dict[str, Any]:
    payload = {
        "chunk_id": row.id,
        "document_id": str(row.document_id),
        "section_id": row.section_id,
        "parent_section_id": row.parent_section_id,
        "chunk_index": row.chunk_index,
        "chunk_type": row.chunk_type,
        "content_type": row.content_type,
        "page_start": row.page_start,
        "page_end": row.page_end,
        "token_count": row.token_count,
        "block_ids": _load_json(row.block_ids_json) or [],
        "table_ids": _load_json(row.table_ids_json) or [],
        "index_version": row.index_version,
    }
    if include_content:
        payload["content"] = row.content
    return payload
