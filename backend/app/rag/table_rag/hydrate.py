"""Hydrate retrieved chunks with full structured tables from SQL."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.config import get_settings
from app.rag.table_rag.ids import parse_table_payload
from app.rag.table_rag.representations import compact_table_prompt_block
from app.rag.table_rag.store import load_tables_by_ids

logger = logging.getLogger(__name__)

_ROW_WINDOW_KINDS = frozenset(
    {"table_row_window", "table_row", "table_card", "table_summary"}
)


def _table_ref(chunk: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    document_id = str(chunk.get("document_id") or "").strip()
    table_id = str(chunk.get("table_id") or "").strip()
    if not table_id:
        payload = parse_table_payload(chunk.get("table_json"))
        if payload:
            table_id = str(payload.table_id or "").strip()
    if not document_id or not table_id:
        return None
    return document_id, table_id


def _collect_row_hints(chunk: Dict[str, Any]) -> List[int]:
    hints: List[int] = []
    chunk_type = str(chunk.get("chunk_type") or "").lower()
    index_kind = str(chunk.get("index_kind") or "").lower()
    if chunk_type not in _ROW_WINDOW_KINDS and index_kind not in _ROW_WINDOW_KINDS:
        return hints
    raw_indices = chunk.get("row_indices")
    if isinstance(raw_indices, (list, tuple)):
        for item in raw_indices:
            try:
                hints.append(int(item))
            except (TypeError, ValueError):
                pass
    raw_idx = chunk.get("row_index")
    if raw_idx is None:
        payload = parse_table_payload(chunk.get("table_json"))
        if payload is not None:
            raw_idx = payload.row_index
            if payload.row_indices:
                for item in payload.row_indices:
                    try:
                        hints.append(int(item))
                    except (TypeError, ValueError):
                        pass
    try:
        if raw_idx is not None:
            hints.append(int(raw_idx))
    except (TypeError, ValueError):
        pass
    return hints


def hydrate_retrieved_tables(
    chunks: Sequence[Dict[str, Any]],
    *,
    user_id: str,
) -> List[Dict[str, Any]]:
    """Attach full ``table`` payloads from ``document_tables`` onto chunks.

    Also sets ``table_prompt_block`` for answer-generation formatting and
    preserves relevant ``row_index`` hints from table_row_window hits.

    Enforces ``index_version`` parity between the retrieved chunk and the SQL
    row when both sides declare a version; mismatches are flagged but still
    hydrate so answer-gen can proceed with a warning.
    """
    if not chunks:
        return []

    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for chunk in chunks:
        ref = _table_ref(chunk)
        if ref and ref not in seen:
            seen.add(ref)
            pairs.append(ref)

    loaded: Dict[str, Dict[str, Any]] = {}
    if pairs:
        try:
            loaded = load_tables_by_ids(user_id=user_id, pairs=pairs)
        except Exception as exc:
            logger.warning("TableRAG hydrate failed: %s", exc)
            loaded = {}

    # Collect relevant row indices per table from row-window hits.
    row_hits: Dict[str, Set[int]] = {}
    for chunk in chunks:
        ref = _table_ref(chunk)
        if not ref:
            continue
        key = f"{ref[0]}:{ref[1]}"
        for idx in _collect_row_hints(chunk):
            row_hits.setdefault(key, set()).add(idx)

    settings = get_settings()
    expected_version = str(
        getattr(settings, "TABLE_RAG_INDEX_VERSION", "") or ""
    ).strip()

    out: List[Dict[str, Any]] = []
    for chunk in chunks:
        item = dict(chunk)
        ref = _table_ref(item)
        if not ref:
            out.append(item)
            continue
        key = f"{ref[0]}:{ref[1]}"
        stored = loaded.get(key)
        if stored and stored.get("table"):
            full_table = dict(stored["table"])
            # Propagate multipage list onto the hydrated grid.
            pages = stored.get("pages") or full_table.get("pages") or []
            if pages and not full_table.get("pages"):
                full_table["pages"] = pages
            item["table"] = full_table
            item["table_id"] = stored["table_id"]
            item["table_json"] = json.dumps(full_table, ensure_ascii=False)
            item["table_needs_verification"] = bool(
                stored.get("needs_verification")
                or full_table.get("needs_verification")
            )
            # index_version parity check
            chunk_version = str(item.get("index_version") or expected_version or "").strip()
            stored_version = str(stored.get("index_version") or "").strip()
            if chunk_version and stored_version and chunk_version != stored_version:
                logger.warning(
                    "TableRAG index_version mismatch for %s:%s chunk=%s sql=%s",
                    ref[0],
                    ref[1],
                    chunk_version,
                    stored_version,
                )
                item["table_index_version_mismatch"] = True
                item["table_needs_verification"] = True
            elif expected_version and stored_version and expected_version != stored_version:
                logger.warning(
                    "TableRAG expected index_version %s but SQL has %s for %s",
                    expected_version,
                    stored_version,
                    key,
                )
                item["table_index_version_mismatch"] = True
            relevant = sorted(row_hits.get(key) or [])
            item["table_prompt_block"] = compact_table_prompt_block(
                full_table,
                max_rows=20,
                relevant_row_indices=relevant or None,
            )
            if relevant:
                item["relevant_row_indices"] = relevant
            if pages:
                item["pages"] = pages
        else:
            # Fallback: use whatever table_json the chunk already carries.
            payload = parse_table_payload(item.get("table_json"))
            if payload:
                table = payload.to_dict()
                item["table"] = table
                item["table_prompt_block"] = compact_table_prompt_block(table)
                if table.get("needs_verification") or table.get("low_confidence_cells"):
                    item["table_needs_verification"] = True
        out.append(item)
    return out
