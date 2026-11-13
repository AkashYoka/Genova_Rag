"""Build table-level / row-level index chunks and extract storeable tables."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.rag.table_rag.ids import (
    ensure_table_id,
    parse_table_payload,
    table_fingerprint,
)
from app.config import get_settings
from app.rag.table_rag.representations import (
    table_interpretation_text,
    table_row_text,
    table_summary_text,
)

logger = logging.getLogger(__name__)

CHUNK_TYPE_TABLE_SUMMARY = "table_summary"
CHUNK_TYPE_TABLE_INTERPRETATION = "table_interpretation"
CHUNK_TYPE_TABLE_ROW = "table_row"
INDEX_KINDS = {
    CHUNK_TYPE_TABLE_SUMMARY,
    CHUNK_TYPE_TABLE_INTERPRETATION,
    CHUNK_TYPE_TABLE_ROW,
}


def _parse_chunk_table(chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    table = parse_table_payload(chunk.get("table_json"))
    if not table:
        return None
    # Prefer chunk-level page when table payload omits it.
    if table.get("page") in (None, "", 0):
        table["page"] = chunk.get("page") or chunk.get("page_start") or 1
    return table


def _is_index_chunk(chunk: Dict[str, Any]) -> bool:
    chunk_type = str(chunk.get("chunk_type") or "").lower()
    index_kind = str(chunk.get("index_kind") or "").lower()
    return chunk_type in INDEX_KINDS or index_kind in INDEX_KINDS


def extract_tables_from_chunks(
    chunks: Sequence[Dict[str, Any]],
    *,
    document_id: str = "",
    filename: str = "",
) -> List[Dict[str, Any]]:
    """Deduplicate full tables from chunk ``table_json`` payloads.

    Split table parts are merged by ``table_id`` / fingerprint, preferring the
    payload with the most rows (closest to the complete extracted table).
    """
    best: Dict[str, Dict[str, Any]] = {}
    ordinal = 0

    for chunk in chunks:
        if _is_index_chunk(chunk):
            continue
        table = _parse_chunk_table(chunk)
        if not table:
            continue
        chunk_type = str(chunk.get("chunk_type") or chunk.get("content_type") or "").lower()
        if chunk_type not in ("table", "") and not chunk.get("table_json"):
            continue

        table_id = str(chunk.get("table_id") or table.get("table_id") or "").strip()
        if not table_id:
            table_id = ensure_table_id(
                table,
                ordinal=ordinal,
                document_id=document_id,
                page_fallback=int(chunk.get("page") or 1),
            )
            ordinal += 1
        else:
            table["table_id"] = table_id

        # When a split part carries fewer rows, keep the richer copy.
        # Prefer the copy that also carries LLM understanding metadata.
        fp = table_fingerprint(table)
        rows = table.get("rows") or []
        existing = best.get(fp)
        has_llm = bool(table.get("llm_understanding"))
        existing_llm = bool((existing or {}).get("table", {}).get("llm_understanding"))
        richer = existing is None or len(rows) > len(
            (existing or {}).get("table", {}).get("rows") or []
        )
        better_meta = existing is not None and len(rows) == len(
            existing["table"].get("rows") or []
        ) and has_llm and not existing_llm
        if richer or better_meta:
            # Layout/fallback chunks may lack LLM understanding — attach a
            # deterministic skeleton so indexes stay consistent.
            if not table.get("llm_understanding"):
                try:
                    from app.rag.table_rag.understanding import (
                        apply_understanding_to_table,
                        deterministic_understanding,
                    )

                    table = apply_understanding_to_table(
                        table,
                        deterministic_understanding(
                            table,
                            section_path=" > ".join(
                                p
                                for p in (
                                    str(chunk.get("section") or ""),
                                    str(chunk.get("subsection") or ""),
                                )
                                if p
                            ),
                        ),
                    )
                except Exception:
                    pass
            separate_interp = bool(
                getattr(
                    get_settings(),
                    "TABLE_RAG_SEPARATE_INTERPRETATION_EMBEDDING",
                    True,
                )
            )
            summary = table_summary_text(
                table,
                filename=filename or str(chunk.get("filename") or ""),
                section=str(chunk.get("section") or ""),
                subsection=str(chunk.get("subsection") or ""),
                include_llm=not separate_interp,
            )
            best[fp] = {
                "table_id": table_id,
                "table": table,
                "page": int(table.get("page") or chunk.get("page") or 1),
                "page_start": int(chunk.get("page_start") or table.get("page") or 1),
                "page_end": int(chunk.get("page_end") or table.get("page") or 1),
                "section": str(chunk.get("section") or ""),
                "subsection": str(chunk.get("subsection") or ""),
                "filename": filename or str(chunk.get("filename") or ""),
                "extraction_method": str(
                    table.get("extraction_method")
                    or chunk.get("extraction_method")
                    or ""
                ),
                "summary_text": summary,
                "llm_understanding": table.get("llm_understanding"),
                "needs_verification": bool(
                    table.get("needs_verification")
                    or table.get("low_confidence_cells")
                    or (
                        isinstance(table.get("structure_validation"), dict)
                        and table["structure_validation"].get("status")
                        != "structure_valid"
                    )
                ),
                "structure_validation": table.get("structure_validation"),
                "ordinal": len(best),
            }

    records = sorted(
        best.values(),
        key=lambda item: (item.get("page_start") or 0, item.get("ordinal") or 0),
    )
    for idx, record in enumerate(records):
        record["ordinal"] = idx
    return records


def _annotate_source_chunks(
    chunks: List[Dict[str, Any]],
    tables: Sequence[Dict[str, Any]],
) -> None:
    """Ensure source table chunks carry ``table_id`` / ``table_json`` with id."""
    by_id = {r["table_id"]: r for r in tables}
    by_fp = {table_fingerprint(r["table"]): r for r in tables}

    for chunk in chunks:
        if _is_index_chunk(chunk):
            continue
        table = _parse_chunk_table(chunk)
        if not table:
            continue
        table_id = str(chunk.get("table_id") or table.get("table_id") or "").strip()
        record = by_id.get(table_id) if table_id else None
        if record is None:
            record = by_fp.get(table_fingerprint(table))
        if record is None:
            continue
        full = dict(record["table"])
        full["table_id"] = record["table_id"]
        # Preserve split-part metadata from the chunk payload when present.
        for key in ("part", "part_row_start", "part_row_count"):
            if key in table:
                full[key] = table[key]
        chunk["table_id"] = record["table_id"]
        chunk["table_ids"] = json.dumps([record["table_id"]])
        chunk["table_json"] = json.dumps(full, ensure_ascii=False)
        if record.get("needs_verification"):
            chunk["table_needs_verification"] = True


def _build_index_chunks(
    tables: Sequence[Dict[str, Any]],
    *,
    document_id: str,
    filename: str,
    index_version: str = "",
    max_rows_per_table: int = 200,
) -> List[Dict[str, Any]]:
    """Create table_summary + table_interpretation + table_row index chunks."""
    index_chunks: List[Dict[str, Any]] = []
    settings = get_settings()
    separate_interp = bool(
        getattr(settings, "TABLE_RAG_SEPARATE_INTERPRETATION_EMBEDDING", True)
    )

    for record in tables:
        table = record["table"]
        table_id = record["table_id"]
        headers = list(table.get("headers") or [])
        rows = [list(r) for r in (table.get("rows") or [])]
        page = int(record.get("page_start") or table.get("page") or 1)
        page_end = int(record.get("page_end") or page)
        section = str(record.get("section") or "")
        subsection = str(record.get("subsection") or "")
        extraction_method = str(record.get("extraction_method") or "")
        needs_verification = bool(record.get("needs_verification"))

        # Compact reference payload — full structure lives in SQL.
        ref_payload = {
            "table_id": table_id,
            "page": page,
            "headers": headers,
            "n_rows": int(table.get("n_rows") or (len(rows) + (1 if headers else 0))),
            "n_cols": int(table.get("n_cols") or len(headers) or 0),
            "caption": table.get("caption") or "",
            "extraction_method": extraction_method,
            "needs_verification": needs_verification,
            "merged_cells": table.get("merged_cells") or [],
            "low_confidence_cells": table.get("low_confidence_cells") or [],
            # Keep a bounded row preview on summary chunks for offline lookup.
            "rows": rows[: min(len(rows), 40)],
        }
        if table.get("llm_understanding"):
            ref_payload["llm_understanding"] = table.get("llm_understanding")

        summary_text = record.get("summary_text") or table_summary_text(
            table,
            filename=filename,
            section=section,
            subsection=subsection,
            include_llm=not separate_interp,
        )
        summary_chunk: Dict[str, Any] = {
            "text": summary_text,
            "page": page,
            "page_start": page,
            "page_end": page_end,
            "chunk_type": CHUNK_TYPE_TABLE_SUMMARY,
            "content_type": "table",
            "index_kind": CHUNK_TYPE_TABLE_SUMMARY,
            "section": section,
            "subsection": subsection,
            "extraction_method": extraction_method,
            "table_id": table_id,
            "table_ids": json.dumps([table_id]),
            "table_json": json.dumps(ref_payload, ensure_ascii=False),
            "filename": filename,
        }
        if document_id:
            summary_chunk["document_id"] = str(document_id)
        if index_version:
            summary_chunk["index_version"] = index_version
        if needs_verification:
            summary_chunk["table_needs_verification"] = True
        index_chunks.append(summary_chunk)

        if separate_interp and table.get("llm_understanding"):
            interp_chunk: Dict[str, Any] = {
                "text": table_interpretation_text(
                    table,
                    filename=filename,
                    section=section,
                    subsection=subsection,
                ),
                "page": page,
                "page_start": page,
                "page_end": page_end,
                "chunk_type": CHUNK_TYPE_TABLE_INTERPRETATION,
                "content_type": "table",
                "index_kind": CHUNK_TYPE_TABLE_INTERPRETATION,
                "section": section,
                "subsection": subsection,
                "extraction_method": extraction_method,
                "table_id": table_id,
                "table_ids": json.dumps([table_id]),
                "table_json": json.dumps(ref_payload, ensure_ascii=False),
                "filename": filename,
            }
            if document_id:
                interp_chunk["document_id"] = str(document_id)
            if index_version:
                interp_chunk["index_version"] = index_version
            if needs_verification:
                interp_chunk["table_needs_verification"] = True
            index_chunks.append(interp_chunk)

        for row_index, row in enumerate(rows[:max_rows_per_table]):
            row_payload = {
                **ref_payload,
                "rows": [list(row)],
                "row_index": row_index,
            }
            row_chunk: Dict[str, Any] = {
                "text": table_row_text(
                    table,
                    row_index,
                    row,
                    filename=filename,
                    section=section,
                ),
                "page": page,
                "page_start": page,
                "page_end": page_end,
                "chunk_type": CHUNK_TYPE_TABLE_ROW,
                "content_type": "table",
                "index_kind": CHUNK_TYPE_TABLE_ROW,
                "row_index": row_index,
                "section": section,
                "subsection": subsection,
                "extraction_method": extraction_method,
                "table_id": table_id,
                "table_ids": json.dumps([table_id]),
                "table_json": json.dumps(row_payload, ensure_ascii=False),
                "filename": filename,
            }
            if document_id:
                row_chunk["document_id"] = str(document_id)
            if index_version:
                row_chunk["index_version"] = index_version
            if needs_verification:
                row_chunk["table_needs_verification"] = True
            index_chunks.append(row_chunk)

    return index_chunks


def expand_table_rag_chunks(
    chunks: Sequence[Dict[str, Any]],
    *,
    document_id: str,
    filename: str = "",
    index_version: str = "",
    enabled: bool = True,
    max_rows_per_table: int = 200,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Annotate table chunks and append table/row index entries.

    Returns ``(expanded_chunks, table_records)``. Chunk indexes are renumbered
    contiguously so Chroma/BM25 ids stay unique. Existing text/table chunks are
    preserved; index chunks are additive.
    """
    base_chunks = [dict(c) for c in chunks]
    if not enabled:
        return base_chunks, []

    # Drop any previously expanded index chunks (safe re-run).
    base_chunks = [c for c in base_chunks if not _is_index_chunk(c)]

    tables = extract_tables_from_chunks(
        base_chunks, document_id=document_id, filename=filename
    )
    if not tables:
        logger.info(
            "TableRAG: no structured tables found for document %s", document_id
        )
        for idx, chunk in enumerate(base_chunks):
            chunk["chunk_index"] = idx
        return base_chunks, []

    _annotate_source_chunks(base_chunks, tables)
    index_chunks = _build_index_chunks(
        tables,
        document_id=document_id,
        filename=filename,
        index_version=index_version,
        max_rows_per_table=max_rows_per_table,
    )

    expanded = base_chunks + index_chunks
    for idx, chunk in enumerate(expanded):
        chunk["chunk_index"] = idx
        if document_id and not chunk.get("document_id"):
            chunk["document_id"] = str(document_id)
        if filename and not chunk.get("filename"):
            chunk["filename"] = filename
        if index_version and not chunk.get("index_version"):
            chunk["index_version"] = index_version

    logger.info(
        "TableRAG: document %s → %d tables, %d index chunks "
        "(%d summary + %d interpretation + %d row), total chunks %d",
        document_id,
        len(tables),
        len(index_chunks),
        sum(1 for c in index_chunks if c.get("chunk_type") == CHUNK_TYPE_TABLE_SUMMARY),
        sum(
            1
            for c in index_chunks
            if c.get("chunk_type") == CHUNK_TYPE_TABLE_INTERPRETATION
        ),
        sum(1 for c in index_chunks if c.get("chunk_type") == CHUNK_TYPE_TABLE_ROW),
        len(expanded),
    )
    return expanded, tables
