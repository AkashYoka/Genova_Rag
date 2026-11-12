"""Document hierarchy, section navigation, index status, and reindex APIs.

The relational hierarchy is a navigation/audit index. Retrieval still uses
ChromaDB, BM25, RRF and the cross-encoder reranker unchanged.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import Document, User
from app.rag.chunking.semantic.hierarchy_store import (
    load_section,
    load_section_chunks,
    load_sections,
    serialize_chunk,
    serialize_section,
)
from app.schemas import (
    DocumentHierarchyResponse,
    DocumentIndexStatusResponse,
    ReindexRequest,
    ReindexResponse,
    RelatedSectionsResponse,
    SectionChunksResponse,
    SectionTablesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["document-hierarchy"])

_ALLOWED_STRATEGIES = {"semantic_llm", "deterministic", "legacy"}


def _get_owned_document(db: Session, document_id: str, user: User) -> Document:
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.user_id == user.id,
            Document.is_deleted.is_(False),
        )
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


def _parse_report(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _tree(rows) -> List[Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        item = serialize_section(row)
        payload[row.id] = item
        order.append(row.id)

    roots: List[Dict[str, Any]] = []
    for section_id in order:
        item = payload[section_id]
        parent_id = item.get("parent_section_id")
        if parent_id and parent_id in payload:
            payload[parent_id]["children"].append(item)
        else:
            item["parent_section_id"] = None
            roots.append(item)
    return roots


def _find_row(rows, section_id: str):
    for row in rows:
        if row.id == section_id:
            return row
    return None


@router.get("/{document_id}/hierarchy", response_model=DocumentHierarchyResponse)
def get_document_hierarchy(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the expandable section tree for a document."""
    doc = _get_owned_document(db, document_id, current_user)
    rows = load_sections(document_id, db)
    return DocumentHierarchyResponse(
        document_id=str(doc.id),
        filename=doc.original_name,
        chunk_strategy=doc.chunk_strategy,
        index_version=doc.index_version,
        section_count=len(rows),
        chunk_count=doc.chunk_count or 0,
        report=_parse_report(doc.chunking_report),
        sections=_tree(rows),
    )


@router.get("/{document_id}/index-status", response_model=DocumentIndexStatusResponse)
def get_document_index_status(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return indexing progress, strategy, version, and the validation report."""
    doc = _get_owned_document(db, document_id, current_user)
    rows = load_sections(document_id, db)
    return DocumentIndexStatusResponse(
        document_id=str(doc.id),
        status=doc.status,
        processing_stage=doc.processing_stage,
        processing_progress=doc.processing_progress,
        error_message=doc.error_message,
        chunk_strategy=doc.chunk_strategy,
        index_version=doc.index_version,
        section_count=len(rows),
        chunk_count=doc.chunk_count or 0,
        report=_parse_report(doc.chunking_report),
    )


@router.get("/{document_id}/sections/{section_id}", response_model=Dict[str, Any])
def get_section(
    document_id: str,
    section_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return one section with its direct children."""
    doc = _get_owned_document(db, document_id, current_user)
    rows = load_sections(str(doc.id), db)
    row = _find_row(rows, section_id)
    if not row:
        raise HTTPException(status_code=404, detail="Section not found")
    children = [serialize_section(child) for child in rows if child.parent_section_id == section_id]
    return serialize_section(row, children=children)


@router.get(
    "/{document_id}/sections/{section_id}/chunks",
    response_model=SectionChunksResponse,
)
def get_section_chunks(
    document_id: str,
    section_id: str,
    include_content: bool = Query(default=True),
    limit: int = Query(default=500, ge=1, le=2000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return persisted chunks belonging directly to a section."""
    doc = _get_owned_document(db, document_id, current_user)
    row = load_section(str(doc.id), section_id, db)
    if not row:
        raise HTTPException(status_code=404, detail="Section not found")
    chunks = load_section_chunks(str(doc.id), section_id, db, limit=limit)
    return SectionChunksResponse(
        document_id=str(doc.id),
        section_id=section_id,
        title=row.title,
        chunk_count=len(chunks),
        chunks=[
            serialize_chunk(chunk, include_content=include_content)
            for chunk in chunks
        ],
    )


@router.get(
    "/{document_id}/sections/{section_id}/tables",
    response_model=SectionTablesResponse,
)
def get_section_tables(
    document_id: str,
    section_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return tables associated with a section (headers, caption, page)."""
    doc = _get_owned_document(db, document_id, current_user)
    row = load_section(str(doc.id), section_id, db)
    if not row:
        raise HTTPException(status_code=404, detail="Section not found")
    tables = serialize_section(row).get("tables") or []
    return SectionTablesResponse(
        document_id=str(doc.id),
        section_id=section_id,
        title=row.title,
        tables=tables,
    )


@router.get(
    "/{document_id}/sections/{section_id}/related",
    response_model=RelatedSectionsResponse,
)
def get_related_sections(
    document_id: str,
    section_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return parent, children, siblings, and previous/next navigation."""
    doc = _get_owned_document(db, document_id, current_user)
    rows = load_sections(str(doc.id), db)
    row = _find_row(rows, section_id)
    if not row:
        raise HTTPException(status_code=404, detail="Section not found")

    siblings = [item for item in rows if item.parent_section_id == row.parent_section_id]
    position = next(
        (index for index, item in enumerate(siblings) if item.id == section_id),
        -1,
    )
    previous_row = siblings[position - 1] if position > 0 else None
    next_row = siblings[position + 1] if 0 <= position < len(siblings) - 1 else None
    parent_row = _find_row(rows, row.parent_section_id) if row.parent_section_id else None
    children = [item for item in rows if item.parent_section_id == section_id]

    return RelatedSectionsResponse(
        document_id=str(doc.id),
        section_id=section_id,
        parent=serialize_section(parent_row) if parent_row else None,
        previous=serialize_section(previous_row) if previous_row else None,
        next=serialize_section(next_row) if next_row else None,
        children=[serialize_section(item) for item in children],
        siblings=[
            serialize_section(item)
            for item in siblings
            if item.id != section_id
        ],
    )


@router.post("/{document_id}/reindex", response_model=ReindexResponse)
def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    payload: Optional[ReindexRequest] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Re-run ingestion to rebuild vectors, BM25, graph edges, and hierarchy.

    The previous index stays live until the new ingestion succeeds:
    ``store_chunks`` removes old Chroma/BM25 rows only when it is ready to
    insert replacements, and the hierarchy rows are replaced in the same run.
    The original PDF is never deleted.
    """
    doc = _get_owned_document(db, document_id, current_user)
    if doc.status in {"processing", "pending"}:
        raise HTTPException(
            status_code=409,
            detail=f"Document is already {doc.status}; wait for ingestion to finish",
        )

    strategy = (payload.strategy if payload and payload.strategy else doc.chunk_strategy)
    strategy = (strategy or "semantic_llm").lower()
    if strategy not in _ALLOWED_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported chunk strategy: {strategy}",
        )

    doc.status = "pending"
    doc.processing_stage = "queued"
    doc.processing_progress = 0
    doc.error_message = None
    doc.last_error_traceback = None
    doc.chunk_strategy = strategy
    db.commit()

    task_id: Optional[str] = None
    try:
        from app.tasks import process_document

        task = process_document.delay(
            document_id=str(doc.id),
            filepath="",
            original_name=doc.original_name,
            user_id=str(current_user.id),
        )
        task_id = getattr(task, "id", None)
    except Exception as exc:  # Celery unavailable → in-process fallback
        logger.warning("Celery reindex queue failed for %s: %s", document_id, exc)
        from app.services.document_ingestion import ingest_document

        if background_tasks is not None:
            background_tasks.add_task(
                ingest_document,
                document_id=str(doc.id),
                filepath="",
                original_name=doc.original_name,
                user_id=str(current_user.id),
            )
            task_id = f"local_{uuid.uuid4().hex}"
        else:
            doc.status = "failed"
            doc.error_message = "Could not queue reindex task"
            db.commit()
            raise HTTPException(status_code=503, detail="Could not queue reindex task")

    return ReindexResponse(
        document_id=str(doc.id),
        status="pending",
        task_id=task_id,
        chunk_strategy=strategy,
        message="Reindex queued; the previous index remains live until it succeeds",
    )
