"""Textual representations for table-level and row-level embeddings."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _cell(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def _header_line(headers: Sequence[str]) -> str:
    return " | ".join(_cell(h) for h in headers)


def _row_line(headers: Sequence[str], row: Sequence[Any]) -> str:
    cells = list(row or [])
    width = max(len(headers), len(cells))
    headers = (list(headers) + [""] * width)[:width]
    cells = (cells + [""] * width)[:width]
    paired = []
    for header, cell in zip(headers, cells):
        h = _cell(header)
        c = _cell(cell)
        if h and c:
            paired.append(f"{h}={c}")
        elif c:
            paired.append(c)
    return " | ".join(paired) if paired else " | ".join(_cell(c) for c in cells)


def _understanding_index_text(table: Dict[str, Any]) -> str:
    """Append LLM interpretation metadata when present (never cell values)."""
    raw = table.get("llm_understanding")
    if not raw:
        return ""
    try:
        from app.rag.table_rag.schema import TableUnderstanding

        if isinstance(raw, TableUnderstanding):
            return raw.to_index_text()
        if isinstance(raw, dict):
            return TableUnderstanding.model_validate(raw).to_index_text()
    except Exception:
        # Best-effort plain dump of search phrases only.
        if isinstance(raw, dict):
            phrases = raw.get("semantic_search_phrases") or []
            desc = str(raw.get("description") or "").strip()
            bits = []
            if desc:
                bits.append(f"description: {desc}")
            if phrases:
                bits.append("search_phrases: " + "; ".join(str(p) for p in phrases[:8]))
            if bits:
                return "LLM TABLE INTERPRETATION (not cell ground truth):\n" + "\n".join(bits)
    return ""


def table_summary_text(
    table: Dict[str, Any],
    *,
    filename: str = "",
    section: str = "",
    subsection: str = "",
    include_llm: bool = False,
) -> str:
    """Dense table-level text for semantic + BM25 matching.

    By default this is *original-structure only* (identity, caption, headers,
    shape, merges, facts). LLM interpretation is a separate embedding when
    ``TABLE_RAG_SEPARATE_INTERPRETATION_EMBEDDING`` is enabled; pass
    ``include_llm=True`` only for legacy combined summaries.
    """
    headers = [str(h or "") for h in (table.get("headers") or [])]
    rows = table.get("rows") or []
    caption = _cell(table.get("caption") or "")
    page = table.get("page") or "?"
    facts = [str(f) for f in (table.get("linearized_facts") or []) if str(f).strip()]
    merged = table.get("merged_cells") or []
    low_conf = table.get("low_confidence_cells") or []
    needs_verification = bool(table.get("needs_verification") or low_conf)

    parts: List[str] = [
        f"TABLE SUMMARY | page {page}",
    ]
    if filename:
        parts.append(f"document: {filename}")
    if section:
        parts.append(f"section: {section}")
    if subsection:
        parts.append(f"subsection: {subsection}")
    if caption:
        parts.append(f"caption: {caption}")
    if headers:
        parts.append(f"headers: {_header_line(headers)}")
    parts.append(
        f"shape: {len(rows)} data rows x {int(table.get('n_cols') or len(headers) or 0)} cols"
    )
    if merged:
        merge_bits = []
        for item in merged[:8]:
            if not isinstance(item, dict):
                continue
            merge_bits.append(
                f"r{item.get('row_index', '?')}c{item.get('col_index', '?')}"
                f" span={item.get('row_span', 1)}x{item.get('col_span', 1)}"
                f" '{_cell(item.get('text'))[:40]}'"
            )
        if merge_bits:
            parts.append("merged_cells: " + "; ".join(merge_bits))
    if needs_verification:
        parts.append(
            f"extraction_warning: low-confidence cells present "
            f"({len(low_conf)}); verify values before use"
        )
    if facts:
        parts.append("table facts:")
        parts.extend(f"- {fact}" for fact in facts[:24])
    else:
        # Fall back to a few paired row previews for lexical recall.
        for idx, row in enumerate(rows[:6]):
            parts.append(f"row[{idx}]: {_row_line(headers, row)}")

    if include_llm:
        llm_text = _understanding_index_text(table)
        if llm_text:
            parts.append("")
            parts.append(llm_text)
    return "\n".join(parts).strip()


def table_interpretation_text(
    table: Dict[str, Any],
    *,
    filename: str = "",
    section: str = "",
    subsection: str = "",
) -> str:
    """Separate embedding text for LLM interpretation metadata only."""
    page = table.get("page") or "?"
    caption = _cell(table.get("caption") or "")
    parts: List[str] = [
        f"TABLE INTERPRETATION | page {page}",
    ]
    if filename:
        parts.append(f"document: {filename}")
    if section:
        parts.append(f"section: {section}")
    if subsection:
        parts.append(f"subsection: {subsection}")
    if caption:
        parts.append(f"caption: {caption}")
    llm_text = _understanding_index_text(table)
    if llm_text:
        parts.append(llm_text)
    else:
        parts.append(
            "LLM TABLE INTERPRETATION (not cell ground truth): "
            "unavailable; use extracted table_json for values."
        )
    return "\n".join(parts).strip()


def table_row_text(
    table: Dict[str, Any],
    row_index: int,
    row: Sequence[Any],
    *,
    filename: str = "",
    section: str = "",
) -> str:
    """Row-level text optimized for exact value / multi-constraint lookup."""
    headers = [str(h or "") for h in (table.get("headers") or [])]
    caption = _cell(table.get("caption") or "")
    page = table.get("page") or "?"
    parts: List[str] = [
        f"TABLE ROW | page {page} | row_index {row_index}",
    ]
    if filename:
        parts.append(f"document: {filename}")
    if section:
        parts.append(f"section: {section}")
    if caption:
        parts.append(f"caption: {caption}")
    if headers:
        parts.append(f"headers: {_header_line(headers)}")
    parts.append(f"values: {_row_line(headers, row)}")
    # Flat cell dump helps BM25 exact matches on numeric cells.
    flat_cells = [_cell(c) for c in (row or []) if _cell(c)]
    if flat_cells:
        parts.append("cells: " + " ".join(flat_cells))
    # Header-aware retrieval cues from LLM metadata (no cell rewriting).
    raw = table.get("llm_understanding")
    if isinstance(raw, dict):
        phrases = [str(p) for p in (raw.get("semantic_search_phrases") or [])[:4] if p]
        axes = [
            str(a.get("name") or a.get("header_match") or "")
            for a in (raw.get("lookup_attributes") or [])[:4]
            if isinstance(a, dict)
        ]
        cues = [c for c in (phrases + axes) if c]
        if cues:
            parts.append("semantic_cues: " + " | ".join(cues))
    return "\n".join(parts).strip()


def compact_table_prompt_block(
    table: Dict[str, Any],
    *,
    max_rows: int = 20,
    relevant_row_indices: Optional[Sequence[int]] = None,
) -> str:
    """Structured prompt appendix with headers, rows, spans, and warnings."""
    headers = [str(h or "") for h in (table.get("headers") or [])]
    rows = [list(r) for r in (table.get("rows") or [])]
    page = table.get("page") or "?"
    caption = _cell(table.get("caption") or "")
    merged = table.get("merged_cells") or []
    low_conf = table.get("low_confidence_cells") or []
    table_id = str(table.get("table_id") or "")

    lines = [f"Structured table (page {page}" + (f", id={table_id}" if table_id else "") + "):"]
    if caption:
        lines.append(f"caption: {caption}")
    if headers:
        lines.append("headers: " + " | ".join(headers))

    if relevant_row_indices:
        keep = sorted({int(i) for i in relevant_row_indices if 0 <= int(i) < len(rows)})
        if keep and len(keep) < len(rows):
            for idx in keep:
                cells = [str(c or "") for c in rows[idx]]
                lines.append(f"row[{idx}]: " + " | ".join(cells))
            lines.append(f"(showing {len(keep)} relevant of {len(rows)} rows)")
        else:
            relevant_row_indices = None

    if not relevant_row_indices:
        shown = rows[:max_rows]
        for idx, row in enumerate(shown):
            cells = [str(c or "") for c in row]
            lines.append(f"row[{idx}]: " + " | ".join(cells))
        if len(rows) > max_rows:
            lines.append(f"... ({len(rows) - max_rows} more rows)")

    if merged:
        lines.append("merged_cells:")
        for item in merged[:12]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"  - r{item.get('row_index')}c{item.get('col_index')} "
                f"span={item.get('row_span', 1)}x{item.get('col_span', 1)} "
                f"text={_cell(item.get('text'))!r}"
            )
    if low_conf or table.get("needs_verification"):
        lines.append(
            "WARNING: table extraction has low-confidence cells; "
            "treat numeric values cautiously."
        )
        for item in (low_conf or [])[:6]:
            if isinstance(item, dict):
                lines.append(
                    f"  - low_conf r{item.get('row')}c{item.get('col')}: "
                    f"{_cell(item.get('text'))!r} "
                    f"(conf={item.get('confidence')})"
                )
    llm_block = _understanding_index_text(table)
    if llm_block:
        lines.append("")
        lines.append(llm_block)
    return "\n".join(lines)
