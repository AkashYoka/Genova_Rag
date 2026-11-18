"""Hydrate retrieved chunks with full structured tables from SQL."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.config import get_settings
from app.rag.table_rag.ids import parse_table_payload
from app.rag.table_rag.independent_rows import independent_rows_prompt_block
from app.rag.table_rag.nxt import compact_table_prompt_block
from app.rag.table_rag.store import load_tables_by_ids

logger = logging.getLogger(__name__)

# Generic float/label refs: "Table 1", "table 2", "Tbl. 3", "Table A"
_TABLE_REF_RE = re.compile(
    r"(?i)\b(?:tables?|tbl\.?)\s*([0-9]{1,3}|[A-Z])\b"
)

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


def _content_tokens(text: str) -> set[str]:
    import re

    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "table",
        "every",
        "row",
        "column",
        "cell",
    }
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (text or "").lower())
    return {t for t in toks if t not in stop}


def _caption_of(item: Dict[str, Any]) -> str:
    table = item.get("table") if isinstance(item.get("table"), dict) else {}
    return str(
        (table or {}).get("caption")
        or item.get("section")
        or item.get("caption")
        or ""
    )


def _table_refs_in_text(text: str) -> Set[str]:
    return {m.group(1).upper() for m in _TABLE_REF_RE.finditer(text or "")}


def _scope_lock_bonus(query: str, caption: str) -> int:
    """Large bonus when query names Table N and caption is that same table."""
    q_refs = _table_refs_in_text(query)
    if not q_refs:
        return 0
    c_refs = _table_refs_in_text(caption)
    if q_refs & c_refs:
        return 100
    # Query locked to a table number that this caption does not carry.
    if c_refs and not (q_refs & c_refs):
        return -50
    return 0


def _stub_table_prompt(table: Dict[str, Any]) -> str:
    caption = str(table.get("caption") or table.get("table_title") or "").strip()
    page = table.get("page") or table.get("page_start")
    bits = ["TABLE STUB"]
    if caption:
        bits.append(caption)
    if page not in (None, ""):
        bits.append(f"page {page}")
    return (
        " — ".join(bits)
        + "\n(Not fully expanded; prefer cell_fact lines from higher-ranked tables.)"
    )


def _build_table_prompt(
    table: Dict[str, Any],
    *,
    query: Optional[str],
    max_rows: int,
    relevant_row_indices: Optional[List[int]],
    slim: bool,
    full_facts: bool,
) -> Tuple[str, Optional[List[int]]]:
    """Assessment-style prompt: independent cell facts + optional grid."""
    if not full_facts:
        return _stub_table_prompt(table), relevant_row_indices or None

    relevant = list(relevant_row_indices or [])
    fact_max = None
    try:
        from app.rag.table_rag.routing import route_query

        route = route_query(query or "")
        structured = bool(route.wants_structured_lookup)
    except Exception:
        structured = False

    n_rows = len(table.get("rows") or [])
    if structured:
        # Pass the full matrix for structured lookups (no row truncation).
        fact_max = max(max_rows, n_rows)
        grid_max = max(max_rows, n_rows)
    else:
        fact_max = max_rows
        grid_max = max_rows

    # Full slim grid (source of truth) + query-near cell facts (atomic match).
    grid = compact_table_prompt_block(
        table,
        max_rows=grid_max,
        relevant_row_indices=None if structured else (relevant or None),
        slim=slim,
        include_understanding=False,
    )
    block = independent_rows_prompt_block(
        table,
        max_rows=fact_max if not structured else None,
        relevant_row_indices=relevant or None,
        query=query,
        include_grid=bool(grid.strip()),
        grid_block=grid,
    )
    return block, relevant or None


def hydrate_retrieved_tables(
    chunks: Sequence[Dict[str, Any]],
    *,
    user_id: str,
    query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Attach full ``table`` payloads from ``document_tables`` onto chunks.

    Builds assessment-style ``table_prompt_block`` cell facts for the top
    caption-overlapping tables only (no domain-specific axis code).
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
    full_n = int(getattr(settings, "TABLE_RAG_FULL_FACT_TABLES", 2) or 2)
    slim = bool(getattr(settings, "TABLE_RAG_PROMPT_SLIM", True))
    max_rows = int(getattr(settings, "TABLE_RAG_PROMPT_MAX_ROWS", 8) or 8)

    # Pass 1: attach table payloads (no prompt yet).
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
            pages = stored.get("pages") or full_table.get("pages") or []
            if pages and not full_table.get("pages"):
                full_table["pages"] = pages
            if stored.get("caption") and not full_table.get("caption"):
                full_table["caption"] = stored["caption"]
            item["table"] = full_table
            item["table_id"] = stored["table_id"]
            item["table_json"] = json.dumps(full_table, ensure_ascii=False)
            item["table_needs_verification"] = bool(
                stored.get("needs_verification")
                or full_table.get("needs_verification")
            )
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
            if relevant and len(relevant) > max_rows:
                relevant = relevant[:max_rows]
            item["_hydrate_relevant"] = relevant
            if pages:
                item["pages"] = pages
        else:
            payload = parse_table_payload(item.get("table_json"))
            if payload:
                table = payload.to_dict()
                item["table"] = table
                item["_hydrate_relevant"] = []
                if table.get("needs_verification") or table.get("low_confidence_cells"):
                    item["table_needs_verification"] = True
        out.append(item)

    # Rank by Table-N scope lock first, then caption↔query token overlap.
    q_refs = _table_refs_in_text(query or "")
    if query and query.strip():
        qtok = _content_tokens(query)
        scored: List[Tuple[int, int, int, Dict[str, Any]]] = []
        for idx, item in enumerate(out):
            cap = _caption_of(item)
            scope = _scope_lock_bonus(query, cap)
            overlap = len(qtok & _content_tokens(cap))
            scored.append((scope, overlap, -idx, item))
        if any(s != 0 or o > 0 for s, o, _, _ in scored):
            scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            out = [item for _, _, _, item in scored]

    # Pass 2: full expand only top-N. When query names Table N, expand matching
    # captions first (and prefer them exclusively when any match exists).
    expand_ids: Set[str] = set()
    scoped_ids: List[str] = []
    if q_refs:
        for item in out:
            tid = str(item.get("table_id") or "").strip()
            if not tid or not isinstance(item.get("table"), dict):
                continue
            if _table_refs_in_text(_caption_of(item)) & q_refs:
                if tid not in scoped_ids:
                    scoped_ids.append(tid)
    prefer = scoped_ids or []
    for tid in prefer:
        if len(expand_ids) >= max(full_n, 1):
            break
        expand_ids.add(tid)
    if not scoped_ids:
        for item in out:
            tid = str(item.get("table_id") or "").strip()
            if not tid or not isinstance(item.get("table"), dict):
                continue
            if len(expand_ids) >= max(full_n, 1):
                break
            expand_ids.add(tid)
    elif len(expand_ids) < max(full_n, 1):
        # Fill remaining slots with non-scoped tables only if budget allows.
        for item in out:
            tid = str(item.get("table_id") or "").strip()
            if not tid or tid in expand_ids or not isinstance(item.get("table"), dict):
                continue
            if len(expand_ids) >= max(full_n, 1):
                break
            expand_ids.add(tid)

    for item in out:
        table = item.get("table") if isinstance(item.get("table"), dict) else None
        if not table:
            item.pop("_hydrate_relevant", None)
            continue
        tid = str(item.get("table_id") or "").strip()
        relevant = list(item.pop("_hydrate_relevant", None) or [])
        block, used_rows = _build_table_prompt(
            table,
            query=query,
            max_rows=max_rows,
            relevant_row_indices=relevant or None,
            slim=slim,
            full_facts=(not tid) or (tid in expand_ids),
        )
        item["table_prompt_block"] = block
        if used_rows:
            item["relevant_row_indices"] = used_rows
    return out
