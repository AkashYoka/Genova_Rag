"""Build table-card / row-window index chunks and extract storeable tables."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.rag.table_rag.ids import (
    ensure_table_id,
    parse_table_payload,
    table_fingerprint,
)
from app.rag.table_rag.representations import (
    table_card_text,
    table_row_window_text,
)
from app.rag.table_rag.schema import coerce_table_dict

logger = logging.getLogger(__name__)

CHUNK_TYPE_TABLE_CARD = "table_card"
CHUNK_TYPE_TABLE_ROW_WINDOW = "table_row_window"
# Legacy aliases kept for already-indexed corpora / tests.
CHUNK_TYPE_TABLE_SUMMARY = CHUNK_TYPE_TABLE_CARD
CHUNK_TYPE_TABLE_INTERPRETATION = "table_interpretation"  # deprecated
CHUNK_TYPE_TABLE_ROW = CHUNK_TYPE_TABLE_ROW_WINDOW
INDEX_KINDS = {
    CHUNK_TYPE_TABLE_CARD,
    CHUNK_TYPE_TABLE_ROW_WINDOW,
    "table_summary",
    "table_interpretation",
    "table_row",
}


def _parse_chunk_table(chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = parse_table_payload(chunk.get("table_json"))
    if not payload:
        return None
    table = payload.to_dict()
    # Prefer chunk-level page when table payload omits it.
    if table.get("page") in (None, "", 0):
        table["page"] = chunk.get("page") or chunk.get("page_start") or 1
    if not table.get("pages"):
        pages = []
        for key in ("page", "page_start", "page_end"):
            try:
                val = chunk.get(key) if key != "page" else table.get("page")
                if val is not None:
                    pages.append(int(val))
            except (TypeError, ValueError):
                pass
        if pages:
            table["pages"] = sorted(set(pages))
    return table


def _is_index_chunk(chunk: Dict[str, Any]) -> bool:
    chunk_type = str(chunk.get("chunk_type") or "").lower()
    index_kind = str(chunk.get("index_kind") or "").lower()
    return chunk_type in INDEX_KINDS or index_kind in INDEX_KINDS


def _merge_table_grids(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    """Union rows from split/partial parts without dropping either side."""
    merged = dict(primary)
    p_rows = [list(r) for r in (primary.get("rows") or [])]
    s_rows = [list(r) for r in (secondary.get("rows") or [])]
    # Prefer longer header set; keep primary headers when equal.
    if len(secondary.get("headers") or []) > len(primary.get("headers") or []):
        merged["headers"] = list(secondary.get("headers") or [])
    # Append rows that are not already present (exact match).
    seen = {tuple(str(c) for c in row) for row in p_rows}
    out_rows = list(p_rows)
    for row in s_rows:
        key = tuple(str(c) for c in row)
        if key not in seen:
            seen.add(key)
            out_rows.append(list(row))
    merged["rows"] = out_rows
    pages: List[int] = []
    for src in (primary, secondary):
        for p in src.get("pages") or []:
            try:
                pages.append(int(p))
            except (TypeError, ValueError):
                pass
        for key in ("page", "page_start", "page_end"):
            try:
                if src.get(key) is not None:
                    pages.append(int(src[key]))
            except (TypeError, ValueError):
                pass
    if pages:
        merged["pages"] = sorted(set(pages))
        merged["page"] = min(merged["pages"])
        merged["page_start"] = min(merged["pages"])
        merged["page_end"] = max(merged["pages"])
    # Prefer LLM understanding when only one side has it.
    if secondary.get("llm_understanding") and not primary.get("llm_understanding"):
        merged["llm_understanding"] = secondary.get("llm_understanding")
    if secondary.get("column_profiles") and not primary.get("column_profiles"):
        merged["column_profiles"] = secondary.get("column_profiles")
    headers = merged.get("headers") or []
    merged["n_rows"] = len(out_rows) + (1 if headers else 0)
    merged["n_cols"] = int(
        merged.get("n_cols") or len(headers) or max((len(r) for r in out_rows), default=0)
    )
    return merged


def extract_tables_from_chunks(
    chunks: Sequence[Dict[str, Any]],
    *,
    document_id: str = "",
    filename: str = "",
) -> List[Dict[str, Any]]:
    """Collect full tables from chunk ``table_json`` payloads.

    Split/partial parts that share ``table_id`` / fingerprint are *merged*
    (row union) rather than dropped by a max-row heuristic.
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

        fp = table_fingerprint(table)
        existing = best.get(fp)
        if existing is not None:
            table = _merge_table_grids(existing["table"], table)
        else:
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
            # Ensure column profiles exist for card text.
            if not table.get("column_profiles"):
                try:
                    from app.rag.table_rag.understanding import profile_full_table_grid
                    from app.rag.table_rag.schema import TablePayload

                    profiles = profile_full_table_grid(TablePayload.model_validate(table))
                    table["column_profiles"] = profiles.get("column_profiles") or []
                except Exception:
                    pass

        summary = table_card_text(
            table,
            filename=filename or str(chunk.get("filename") or ""),
            section=str(chunk.get("section") or (existing or {}).get("section") or ""),
            subsection=str(
                chunk.get("subsection") or (existing or {}).get("subsection") or ""
            ),
        )
        pages = table.get("pages") or []
        page_start = int(
            (pages[0] if pages else None)
            or chunk.get("page_start")
            or table.get("page")
            or 1
        )
        page_end = int(
            (pages[-1] if pages else None)
            or chunk.get("page_end")
            or table.get("page")
            or page_start
        )
        best[fp] = {
            "table_id": table_id,
            "table": table,
            "page": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "pages": pages or [page_start],
            "section": str(chunk.get("section") or (existing or {}).get("section") or ""),
            "subsection": str(
                chunk.get("subsection") or (existing or {}).get("subsection") or ""
            ),
            "filename": filename or str(chunk.get("filename") or ""),
            "extraction_method": str(
                table.get("extraction_engine")
                or table.get("extraction_method")
                or chunk.get("extraction_method")
                or ""
            ),
            "confidence": (
                table.get("structure_confidence")
                if table.get("structure_confidence") is not None
                else table.get("ocr_confidence")
                if table.get("ocr_confidence") is not None
                else table.get("textract_confidence")
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
            "ordinal": (existing or {}).get("ordinal", len(best)),
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


def _iter_row_windows(
    rows: Sequence[Sequence[Any]],
    *,
    window_size: int,
    max_windows: int,
) -> List[Tuple[List[int], List[List[Any]]]]:
    """Sliding/grouped windows of ``window_size`` rows (step ≈ window/2)."""
    n = len(rows)
    if n <= 0:
        return []
    size = max(5, min(10, int(window_size or 8)))
    step = max(1, size // 2)
    windows: List[Tuple[List[int], List[List[Any]]]] = []
    start = 0
    while start < n and len(windows) < max_windows:
        end = min(n, start + size)
        indices = list(range(start, end))
        windows.append((indices, [list(rows[i]) for i in indices]))
        if end >= n:
            break
        start += step
    return windows


def _build_index_chunks(
    tables: Sequence[Dict[str, Any]],
    *,
    document_id: str,
    filename: str,
    index_version: str = "",
    max_rows_per_table: int = 200,
) -> List[Dict[str, Any]]:
    """Create table_card (+ table_row_window for large tables) index chunks."""
    index_chunks: List[Dict[str, Any]] = []
    settings = get_settings()
    window_threshold = int(getattr(settings, "TABLE_RAG_ROW_WINDOW_THRESHOLD", 10) or 10)
    window_size = int(getattr(settings, "TABLE_RAG_ROW_WINDOW_SIZE", 8) or 8)
    max_windows = int(getattr(settings, "TABLE_RAG_MAX_ROW_WINDOWS", 40) or 40)

    for record in tables:
        table = coerce_table_dict(record["table"])
        table_id = record["table_id"]
        headers = list(table.get("headers") or [])
        rows = [list(r) for r in (table.get("rows") or [])][:max_rows_per_table]
        pages = list(record.get("pages") or table.get("pages") or [])
        page = int(record.get("page_start") or table.get("page") or (pages[0] if pages else 1))
        page_end = int(record.get("page_end") or (pages[-1] if pages else page))
        if not pages:
            pages = list(range(page, page_end + 1)) if page_end >= page else [page]
        section = str(record.get("section") or "")
        subsection = str(record.get("subsection") or "")
        extraction_method = str(record.get("extraction_method") or "")
        needs_verification = bool(record.get("needs_verification"))

        # Compact reference payload — full structure lives in SQL.
        ref_payload = {
            "table_id": table_id,
            "page": page,
            "pages": pages,
            "page_start": page,
            "page_end": page_end,
            "headers": headers,
            "n_rows": int(table.get("n_rows") or (len(rows) + (1 if headers else 0))),
            "n_cols": int(table.get("n_cols") or len(headers) or 0),
            "caption": table.get("caption") or "",
            "extraction_method": extraction_method,
            "needs_verification": needs_verification,
            "merged_cells": table.get("merged_cells") or [],
            "low_confidence_cells": table.get("low_confidence_cells") or [],
            "column_profiles": table.get("column_profiles") or [],
            # Bounded preview only — hydrate from SQL for full grids.
            "rows": rows[: min(len(rows), 40)],
        }
        if table.get("llm_understanding"):
            ref_payload["llm_understanding"] = table.get("llm_understanding")
        if table.get("normalized_header_paths"):
            ref_payload["normalized_header_paths"] = table.get("normalized_header_paths")

        card_text = record.get("summary_text") or table_card_text(
            table,
            filename=filename,
            section=section,
            subsection=subsection,
        )
        card_chunk: Dict[str, Any] = {
            "text": card_text,
            "page": page,
            "page_start": page,
            "page_end": page_end,
            "chunk_type": CHUNK_TYPE_TABLE_CARD,
            "content_type": "table",
            "index_kind": CHUNK_TYPE_TABLE_CARD,
            "section": section,
            "subsection": subsection,
            "extraction_method": extraction_method,
            "table_id": table_id,
            "table_ids": json.dumps([table_id]),
            "table_json": json.dumps(ref_payload, ensure_ascii=False),
            "filename": filename,
        }
        if document_id:
            card_chunk["document_id"] = str(document_id)
        if index_version:
            card_chunk["index_version"] = index_version
        if needs_verification:
            card_chunk["table_needs_verification"] = True
        index_chunks.append(card_chunk)

        if len(rows) > window_threshold:
            for indices, window_rows in _iter_row_windows(
                rows, window_size=window_size, max_windows=max_windows
            ):
                window_payload = {
                    **ref_payload,
                    "rows": window_rows,
                    "row_indices": indices,
                    "row_index": indices[0] if indices else 0,
                }
                window_chunk: Dict[str, Any] = {
                    "text": table_row_window_text(
                        table,
                        indices,
                        window_rows,
                        filename=filename,
                        section=section,
                    ),
                    "page": page,
                    "page_start": page,
                    "page_end": page_end,
                    "chunk_type": CHUNK_TYPE_TABLE_ROW_WINDOW,
                    "content_type": "table",
                    "index_kind": CHUNK_TYPE_TABLE_ROW_WINDOW,
                    "row_index": indices[0] if indices else 0,
                    "row_indices": indices,
                    "section": section,
                    "subsection": subsection,
                    "extraction_method": extraction_method,
                    "table_id": table_id,
                    "table_ids": json.dumps([table_id]),
                    "table_json": json.dumps(window_payload, ensure_ascii=False),
                    "filename": filename,
                }
                if document_id:
                    window_chunk["document_id"] = str(document_id)
                if index_version:
                    window_chunk["index_version"] = index_version
                if needs_verification:
                    window_chunk["table_needs_verification"] = True
                index_chunks.append(window_chunk)

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
    """Annotate table chunks and append table_card / row_window index entries.

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
    _table_chunk_types = {
        CHUNK_TYPE_TABLE_CARD,
        CHUNK_TYPE_TABLE_ROW_WINDOW,
        "table",
        "table_summary",
        "table_row",
        "table_interpretation",
    }
    for idx, chunk in enumerate(expanded):
        chunk["chunk_index"] = idx
        if document_id and not chunk.get("document_id"):
            chunk["document_id"] = str(document_id)
        if filename and not chunk.get("filename"):
            chunk["filename"] = filename
        if index_version:
            ctype = str(chunk.get("chunk_type") or chunk.get("index_kind") or "")
            # Force TableRAG version onto every table-bearing chunk so hydrate
            # parity matches document_tables.index_version (not semantic-llm-*).
            if (
                ctype in _table_chunk_types
                or chunk.get("table_id")
                or chunk.get("table_json")
            ):
                chunk["index_version"] = index_version
            elif not chunk.get("index_version"):
                chunk["index_version"] = index_version

    n_cards = sum(
        1
        for c in index_chunks
        if c.get("chunk_type") in (CHUNK_TYPE_TABLE_CARD, "table_summary")
    )
    n_windows = sum(
        1
        for c in index_chunks
        if c.get("chunk_type") in (CHUNK_TYPE_TABLE_ROW_WINDOW, "table_row")
    )
    logger.info(
        "TableRAG: document %s → %d tables, %d index chunks "
        "(%d table_card + %d table_row_window), total chunks %d",
        document_id,
        len(tables),
        len(index_chunks),
        n_cards,
        n_windows,
        len(expanded),
    )
    return expanded, tables
