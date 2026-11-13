"""Hydrate retrieved chunks with full structured tables from SQL."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.rag.table_rag.ids import parse_table_payload
from app.rag.table_rag.representations import compact_table_prompt_block
from app.rag.table_rag.store import load_tables_by_ids

logger = logging.getLogger(__name__)


def _table_ref(chunk: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    document_id = str(chunk.get("document_id") or "").strip()
    table_id = str(chunk.get("table_id") or "").strip()
    if not table_id:
        table = parse_table_payload(chunk.get("table_json"))
        if table:
            table_id = str(table.get("table_id") or "").strip()
    if not document_id or not table_id:
        return None
    return document_id, table_id


def hydrate_retrieved_tables(
    chunks: Sequence[Dict[str, Any]],
    *,
    user_id: str,
) -> List[Dict[str, Any]]:
    """Attach full ``table`` payloads from ``document_tables`` onto chunks.

    Also sets ``table_prompt_block`` for answer-generation formatting and
    preserves relevant ``row_index`` hints from table_row hits.
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

    # Collect relevant row indices per table from row-level hits.
    row_hits: Dict[str, Set[int]] = {}
    for chunk in chunks:
        ref = _table_ref(chunk)
        if not ref:
            continue
        key = f"{ref[0]}:{ref[1]}"
        chunk_type = str(chunk.get("chunk_type") or "").lower()
        if chunk_type == "table_row" or chunk.get("index_kind") == "table_row":
            raw_idx = chunk.get("row_index")
            if raw_idx is None:
                table = parse_table_payload(chunk.get("table_json")) or {}
                raw_idx = table.get("row_index")
            try:
                row_hits.setdefault(key, set()).add(int(raw_idx))
            except (TypeError, ValueError):
                pass

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
            item["table"] = full_table
            item["table_id"] = stored["table_id"]
            item["table_json"] = json.dumps(full_table, ensure_ascii=False)
            item["table_needs_verification"] = bool(
                stored.get("needs_verification")
                or full_table.get("needs_verification")
            )
            # Prefer SQL page/caption metadata when richer.
            if stored.get("caption") and not item.get("section"):
                pass
            relevant = sorted(row_hits.get(key) or [])
            item["table_prompt_block"] = compact_table_prompt_block(
                full_table,
                max_rows=20,
                relevant_row_indices=relevant or None,
            )
            if relevant:
                item["relevant_row_indices"] = relevant
        else:
            # Fallback: use whatever table_json the chunk already carries.
            table = parse_table_payload(item.get("table_json"))
            if table:
                item["table"] = table
                item["table_prompt_block"] = compact_table_prompt_block(table)
                if table.get("needs_verification") or table.get("low_confidence_cells"):
                    item["table_needs_verification"] = True
        out.append(item)
    return out
