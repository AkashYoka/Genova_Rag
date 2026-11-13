"""SQL persistence for complete extracted table structures."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _json_dump(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_load(raw: Any, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def _row_to_dict(row: Any) -> Dict[str, Any]:
    table_json = _json_load(row.table_json, default={}) or {}
    if not isinstance(table_json, dict):
        table_json = {}
    # Prefer live columns over nested payload for identity fields.
    table_json.setdefault("table_id", row.table_id)
    table_json.setdefault("page", row.page_start)
    table_json.setdefault("headers", _json_load(row.headers_json, default=[]) or [])
    table_json.setdefault("rows", _json_load(row.rows_json, default=[]) or [])
    table_json.setdefault("bbox", _json_load(row.bbox_json, default=None))
    table_json.setdefault("merged_cells", _json_load(row.merged_cells_json, default=[]) or [])
    table_json.setdefault("cells", _json_load(row.cells_json, default=[]) or [])
    table_json.setdefault(
        "header_rows", _json_load(row.header_rows_json, default=[]) or []
    )
    table_json.setdefault(
        "low_confidence_cells",
        _json_load(row.low_confidence_json, default=[]) or [],
    )
    # Always prefer the SQL caption column — Chroma/table_json may still hold
    # an empty string from pre-recovery extracts.
    if row.caption:
        table_json["caption"] = row.caption
    else:
        table_json.setdefault("caption", "")
    table_json.setdefault("extraction_method", row.extraction_method or "")
    pages = _json_load(getattr(row, "pages_json", None), default=None)
    if not isinstance(pages, list) or not pages:
        pages = table_json.get("pages") or []
        if not pages:
            pages = [
                p
                for p in (row.page_start, row.page_end)
                if p is not None
            ]
            pages = sorted(set(int(p) for p in pages)) if pages else []
    if pages and not table_json.get("pages"):
        table_json["pages"] = pages
    llm_understanding = _json_load(
        getattr(row, "llm_understanding_json", None), default=None
    )
    if llm_understanding and not table_json.get("llm_understanding"):
        table_json["llm_understanding"] = llm_understanding
    table_json["needs_verification"] = bool(row.needs_verification)
    table_json["n_rows"] = int(row.n_rows or table_json.get("n_rows") or 0)
    table_json["n_cols"] = int(row.n_cols or table_json.get("n_cols") or 0)
    return {
        "id": row.id,
        "table_id": row.table_id,
        "document_id": str(row.document_id),
        "user_id": str(row.user_id),
        "filename": row.filename or "",
        "page_start": row.page_start,
        "page_end": row.page_end,
        "pages": pages,
        "section": row.section or "",
        "subsection": row.subsection or "",
        "caption": row.caption or "",
        "extraction_method": row.extraction_method or "",
        "n_rows": row.n_rows,
        "n_cols": row.n_cols,
        "needs_verification": bool(row.needs_verification),
        "confidence": float(row.confidence or 0.0),
        "index_version": row.index_version or "",
        "table": table_json,
        "summary_text": row.summary_text or "",
        "llm_understanding": llm_understanding or table_json.get("llm_understanding"),
    }


def persist_document_tables(
    *,
    document_id: str,
    user_id: str,
    filename: str,
    tables: Sequence[Dict[str, Any]],
    index_version: str = "",
    db: Any = None,
) -> int:
    """Replace all stored tables for a document with the provided records.

    Each item in ``tables`` should include at least ``table_id`` and a full
    ``table`` / ``table_json`` payload (headers, rows, spans, bbox, …).
    """
    from app.models import DocumentTable

    owns_session = db is None
    if owns_session:
        from app.database import SessionLocal

        db = SessionLocal()

    try:
        db.query(DocumentTable).filter(
            DocumentTable.document_id == document_id
        ).delete(synchronize_session=False)

        stored = 0
        for ordinal, record in enumerate(tables):
            table = dict(record.get("table") or record.get("table_json") or {})
            if isinstance(record.get("table_json"), str):
                table = _json_load(record["table_json"], default={}) or {}
            table_id = str(record.get("table_id") or table.get("table_id") or "").strip()
            if not table_id:
                logger.warning(
                    "Skipping table without table_id for document %s (ordinal %s)",
                    document_id,
                    ordinal,
                )
                continue

            headers = list(table.get("headers") or record.get("headers") or [])
            rows = [list(r) for r in (table.get("rows") or record.get("rows") or [])]
            pages_raw = record.get("pages") or table.get("pages") or []
            pages: List[int] = []
            for p in pages_raw:
                try:
                    pages.append(int(p))
                except (TypeError, ValueError):
                    pass
            page_start = int(
                record.get("page_start")
                or (min(pages) if pages else None)
                or table.get("page")
                or record.get("page")
                or 1
            )
            page_end = int(
                record.get("page_end")
                or (max(pages) if pages else None)
                or page_start
            )
            if not pages:
                pages = list(range(page_start, page_end + 1)) if page_end >= page_start else [page_start]
            table["pages"] = pages
            table["page_start"] = page_start
            table["page_end"] = page_end
            low_conf = list(
                table.get("low_confidence_cells")
                or record.get("low_confidence_cells")
                or []
            )
            needs_verification = bool(
                table.get("needs_verification")
                or record.get("needs_verification")
                or low_conf
            )
            # Prefer structure confidence (0–1), then OCR/textract (0–100 or 0–1).
            conf_raw = (
                record.get("confidence")
                if record.get("confidence") is not None
                else None
            )
            if conf_raw is None:
                for key in (
                    "structure_confidence",
                    "ocr_confidence",
                    "textract_confidence",
                ):
                    if table.get(key) is not None:
                        conf_raw = table.get(key)
                        break
            try:
                confidence = float(conf_raw) if conf_raw is not None else (
                    0.0 if needs_verification else 1.0
                )
            except (TypeError, ValueError):
                confidence = 0.0 if needs_verification else 1.0
            # Normalize 0–100 OCR scores to 0–1 for SQL storage.
            if confidence > 1.0:
                confidence = confidence / 100.0
            confidence = max(0.0, min(1.0, confidence))

            row = DocumentTable(
                id=f"{document_id}:{table_id}"[:160],
                document_id=document_id,
                user_id=user_id,
                table_id=table_id,
                filename=filename or record.get("filename") or "",
                page_start=page_start,
                page_end=page_end,
                pages_json=_json_dump(pages),
                section=str(record.get("section") or "")[:512],
                subsection=str(record.get("subsection") or "")[:512],
                caption=str(table.get("caption") or record.get("caption") or ""),
                extraction_method=str(
                    table.get("extraction_engine")
                    or table.get("extraction_method")
                    or record.get("extraction_method")
                    or ""
                )[:64],
                n_rows=int(table.get("n_rows") or (len(rows) + (1 if headers else 0))),
                n_cols=int(table.get("n_cols") or len(headers) or 0),
                headers_json=_json_dump(headers),
                rows_json=_json_dump(rows),
                bbox_json=_json_dump(table.get("bbox")),
                merged_cells_json=_json_dump(table.get("merged_cells") or []),
                cells_json=_json_dump(table.get("cells") or []),
                header_rows_json=_json_dump(table.get("header_rows") or []),
                low_confidence_json=_json_dump(low_conf),
                table_json=_json_dump(table),
                summary_text=str(record.get("summary_text") or "")[:65000],
                llm_understanding_json=_json_dump(
                    table.get("llm_understanding")
                    or record.get("llm_understanding")
                ),
                needs_verification=needs_verification,
                confidence=confidence,
                index_version=index_version or None,
                ordinal=int(record.get("ordinal", ordinal)),
                created_at=datetime.now(timezone.utc),
            )
            db.add(row)
            stored += 1

        db.commit()
        logger.info(
            "Persisted %d document_tables for document %s",
            stored,
            document_id,
        )
        return stored
    except Exception:
        db.rollback()
        raise
    finally:
        if owns_session:
            db.close()


def delete_document_tables(document_id: str, db: Any = None) -> int:
    """Remove stored tables for a document (e.g. on hard delete)."""
    from app.models import DocumentTable

    owns_session = db is None
    if owns_session:
        from app.database import SessionLocal

        db = SessionLocal()
    try:
        deleted = (
            db.query(DocumentTable)
            .filter(DocumentTable.document_id == document_id)
            .delete(synchronize_session=False)
        )
        db.commit()
        return int(deleted or 0)
    except Exception:
        db.rollback()
        raise
    finally:
        if owns_session:
            db.close()


def get_document_table(
    *,
    document_id: str,
    table_id: str,
    user_id: Optional[str] = None,
    db: Any = None,
) -> Optional[Dict[str, Any]]:
    """Load one stored table by document + table_id."""
    from app.models import DocumentTable

    owns_session = db is None
    if owns_session:
        from app.database import SessionLocal

        db = SessionLocal()
    try:
        query = db.query(DocumentTable).filter(
            DocumentTable.document_id == document_id,
            DocumentTable.table_id == table_id,
        )
        if user_id:
            query = query.filter(DocumentTable.user_id == user_id)
        row = query.first()
        return _row_to_dict(row) if row else None
    finally:
        if owns_session:
            db.close()


def list_document_tables(
    *,
    document_id: str,
    user_id: Optional[str] = None,
    db: Any = None,
) -> List[Dict[str, Any]]:
    """List all stored tables for a document (ordered by page/ordinal)."""
    from app.models import DocumentTable

    owns_session = db is None
    if owns_session:
        from app.database import SessionLocal

        db = SessionLocal()
    try:
        query = db.query(DocumentTable).filter(
            DocumentTable.document_id == document_id
        )
        if user_id:
            query = query.filter(DocumentTable.user_id == user_id)
        rows = (
            query.order_by(DocumentTable.page_start, DocumentTable.ordinal).all()
        )
        return [_row_to_dict(row) for row in rows]
    finally:
        if owns_session:
            db.close()


def load_tables_by_ids(
    *,
    user_id: str,
    pairs: Sequence[tuple],
    db: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Batch-load tables keyed by ``document_id:table_id``.

    ``pairs`` is a sequence of ``(document_id, table_id)``.
    """
    from app.models import DocumentTable

    if not pairs:
        return {}

    owns_session = db is None
    if owns_session:
        from app.database import SessionLocal

        db = SessionLocal()
    try:
        doc_ids = {str(d) for d, _ in pairs if d}
        table_ids = {str(t) for _, t in pairs if t}
        if not doc_ids or not table_ids:
            return {}
        rows = (
            db.query(DocumentTable)
            .filter(
                DocumentTable.user_id == user_id,
                DocumentTable.document_id.in_(list(doc_ids)),
                DocumentTable.table_id.in_(list(table_ids)),
            )
            .all()
        )
        out: Dict[str, Dict[str, Any]] = {}
        wanted = {(str(d), str(t)) for d, t in pairs}
        for row in rows:
            key_pair = (str(row.document_id), str(row.table_id))
            if key_pair not in wanted:
                continue
            out[f"{row.document_id}:{row.table_id}"] = _row_to_dict(row)
        return out
    finally:
        if owns_session:
            db.close()
