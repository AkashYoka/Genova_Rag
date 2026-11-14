"""Textual representations for table-card and row-window embeddings."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from app.rag.table_rag.schema import TablePayload, coerce_table_dict


def _as_dict(table: Union[Dict[str, Any], TablePayload]) -> Dict[str, Any]:
    return coerce_table_dict(table)


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


def _leaf_headers(table: Dict[str, Any]) -> List[str]:
    paths = table.get("normalized_header_paths") or []
    headers = [str(h or "") for h in (table.get("headers") or [])]
    if paths and len(paths) >= len(headers) and headers:
        leaves: List[str] = []
        for path in paths[: len(headers)]:
            if isinstance(path, (list, tuple)) and path:
                leaves.append(str(path[-1] or ""))
            else:
                leaves.append(str(path or ""))
        if any(leaves):
            return leaves
    return headers


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
        if isinstance(raw, dict):
            phrases = raw.get("semantic_search_phrases") or []
            desc = str(raw.get("description") or "").strip()
            bits = []
            if desc:
                bits.append(f"description: {desc}")
            if phrases:
                bits.append("search_phrases: " + "; ".join(str(p) for p in phrases[:8]))
            dims = raw.get("differentiating_dimensions") or []
            if dims:
                bits.append(
                    "differentiating_dimensions: "
                    + "; ".join(str(d) for d in dims[:8])
                )
            footnotes = raw.get("footnotes") or []
            if footnotes:
                bits.append(
                    "footnotes: "
                    + "; ".join(
                        f"{f.get('marker', '')} {f.get('explanation', '')}".strip()
                        for f in footnotes[:6]
                        if isinstance(f, dict)
                    )
                )
            if bits:
                return "LLM TABLE INTERPRETATION (not cell ground truth):\n" + "\n".join(bits)
    return ""


def _column_profile_lines(table: Dict[str, Any], *, max_cols: int = 24) -> List[str]:
    profiles = table.get("column_profiles") or []
    lines: List[str] = []
    for profile in profiles[:max_cols]:
        if not isinstance(profile, dict):
            try:
                profile = profile.model_dump()  # type: ignore[attr-defined]
            except Exception:
                continue
        header = _cell(profile.get("header") or f"col{profile.get('col_index', '?')}")
        kind = str(profile.get("kind") or "text")
        bits = [f"{header}[{kind}]"]
        if profile.get("min_value") is not None or profile.get("max_value") is not None:
            bits.append(f"min={profile.get('min_value')} max={profile.get('max_value')}")
        units = profile.get("units") or []
        if units:
            bits.append("units=" + ",".join(str(u) for u in units[:4]))
        ops = profile.get("boundary_operators") or []
        if ops:
            bits.append("bounds=" + ",".join(str(o) for o in ops[:6]))
        distinct = profile.get("distinct_values") or []
        if distinct:
            bits.append(
                "values=" + " | ".join(_cell(v) for v in distinct[:8])
            )
        nulls = int(profile.get("null_or_empty_count") or 0)
        if nulls:
            bits.append(f"nulls={nulls}")
        lines.append("  - " + "; ".join(bits))
    return lines


def table_card_text(
    table: Union[Dict[str, Any], TablePayload],
    *,
    filename: str = "",
    section: str = "",
    subsection: str = "",
) -> str:
    """Single table-card embedding: structure + understanding + column profiles."""
    data = _as_dict(table)
    headers = _leaf_headers(data)
    rows = data.get("rows") or []
    caption = _cell(data.get("caption") or "")
    pages = data.get("pages") or []
    page = data.get("page") or (pages[0] if pages else "?")
    page_label = (
        ",".join(str(p) for p in pages)
        if pages and len(pages) > 1
        else str(page)
    )
    facts = [str(f) for f in (data.get("linearized_facts") or []) if str(f).strip()]
    merged = data.get("merged_cells") or []
    low_conf = data.get("low_confidence_cells") or []
    needs_verification = bool(data.get("needs_verification") or low_conf)

    parts: List[str] = [
        f"TABLE CARD | pages {page_label}",
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
        f"shape: {len(rows)} data rows x {int(data.get('n_cols') or len(headers) or 0)} cols"
    )
    profile_lines = _column_profile_lines(data)
    if profile_lines:
        parts.append("column_profiles:")
        parts.extend(profile_lines)
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
        for idx, row in enumerate(rows[:6]):
            parts.append(f"row[{idx}]: {_row_line(headers, row)}")

    llm_text = _understanding_index_text(data)
    if llm_text:
        parts.append("")
        parts.append(llm_text)
    return "\n".join(parts).strip()


def table_row_window_text(
    table: Union[Dict[str, Any], TablePayload],
    row_indices: Sequence[int],
    rows: Sequence[Sequence[Any]],
    *,
    filename: str = "",
    section: str = "",
) -> str:
    """Sliding/grouped row-window text prefixed with reconstructed leaf headers."""
    data = _as_dict(table)
    headers = _leaf_headers(data)
    caption = _cell(data.get("caption") or "")
    pages = data.get("pages") or []
    page = data.get("page") or (pages[0] if pages else "?")
    indices = [int(i) for i in row_indices]
    start = indices[0] if indices else 0
    end = indices[-1] if indices else start
    parts: List[str] = [
        f"TABLE ROW WINDOW | page {page} | rows {start}-{end}",
    ]
    if filename:
        parts.append(f"document: {filename}")
    if section:
        parts.append(f"section: {section}")
    if caption:
        parts.append(f"caption: {caption}")
    if headers:
        parts.append(f"headers: {_header_line(headers)}")
    for idx, row in zip(indices, rows):
        parts.append(f"row[{idx}]: {_row_line(headers, row)}")
        flat_cells = [_cell(c) for c in (row or []) if _cell(c)]
        if flat_cells:
            parts.append(f"cells[{idx}]: " + " ".join(flat_cells))
    raw = data.get("llm_understanding")
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


def render_indexed_table_for_selector(
    table: Union[Dict[str, Any], TablePayload, Any],
) -> str:
    """Render a SQL-hydrated table with explicit [row_idx, col_idx] coordinates.

    Accepts ``DocumentTable``-like records (``.table`` / dict) or raw payloads.
    """
    data: Dict[str, Any]
    meta: Dict[str, Any] = {}
    if hasattr(table, "table") and not isinstance(table, dict):
        # DocumentTable ORM or store dict-like record
        nested = getattr(table, "table", None)
        if isinstance(nested, dict):
            data = dict(nested)
        else:
            data = _as_dict(nested) if nested is not None else {}
        meta = {
            "table_id": getattr(table, "table_id", None) or data.get("table_id"),
            "pages": getattr(table, "pages", None),
            "page_start": getattr(table, "page_start", None),
            "page_end": getattr(table, "page_end", None),
            "section": getattr(table, "section", None),
            "caption": getattr(table, "caption", None),
        }
    elif isinstance(table, dict) and "table" in table and isinstance(table.get("table"), dict):
        data = dict(table["table"])
        meta = {
            "table_id": table.get("table_id") or data.get("table_id"),
            "pages": table.get("pages") or data.get("pages"),
            "page_start": table.get("page_start"),
            "page_end": table.get("page_end"),
            "section": table.get("section"),
            "caption": table.get("caption") or data.get("caption"),
        }
    else:
        data = _as_dict(table)
        meta = {}

    headers = _leaf_headers(data)
    rows = [list(r) for r in (data.get("rows") or [])]
    table_id = str(meta.get("table_id") or data.get("table_id") or "")
    pages = meta.get("pages") or data.get("pages") or []
    if not pages:
        for key in ("page_start", "page", "page_end"):
            val = meta.get(key) if key in meta else data.get(key)
            if val is not None:
                try:
                    pages.append(int(val))
                except (TypeError, ValueError):
                    pass
        pages = sorted(set(pages))
    section = str(
        meta.get("section")
        or data.get("section")
        or ""
    )
    caption = _cell(meta.get("caption") or data.get("caption") or "")
    understanding = data.get("llm_understanding") or {}
    dims = []
    footnotes = []
    purpose = ""
    subject = ""
    app_cues: List[str] = []
    app_attrs: List[str] = []
    if isinstance(understanding, dict):
        dims = [str(d) for d in (understanding.get("differentiating_dimensions") or []) if d]
        footnotes = understanding.get("footnotes") or []
        purpose = str(understanding.get("purpose") or "").strip()
        subject = str(understanding.get("subject") or "").strip()
        app_cues = [
            str(c).strip()
            for c in (understanding.get("applicability_cues") or [])
            if str(c).strip()
        ]
        for attr in understanding.get("applicability_attributes") or []:
            if isinstance(attr, dict):
                key = str(attr.get("key") or "").strip()
                val = str(attr.get("value") or "").strip()
                if key or val:
                    app_attrs.append(f"{key}={val}" if key else val)
            elif attr:
                app_attrs.append(str(attr))

    lines: List[str] = [
        f"table_id: {table_id or 'unknown'}",
        f"pages: {pages or [data.get('page')]}",
    ]
    if section:
        lines.append(f"section_path: {section}")
    if caption:
        lines.append(f"caption: {caption}")
    if purpose:
        lines.append(f"purpose: {purpose}")
    if subject:
        lines.append(f"subject: {subject}")
    if app_cues:
        lines.append("applicability: " + "; ".join(app_cues[:12]))
    if app_attrs:
        lines.append("applicability_attrs: " + "; ".join(app_attrs[:12]))
    if dims:
        lines.append("differentiating_dimensions: " + "; ".join(dims))
    if footnotes:
        bits = []
        for item in footnotes[:8]:
            if isinstance(item, dict):
                bits.append(
                    f"{item.get('marker', '')} {item.get('explanation', '')}".strip()
                )
            else:
                bits.append(str(item))
        if bits:
            lines.append("footnotes: " + " | ".join(bits))
    if headers:
        lines.append("headers: " + " | ".join(headers))
    lines.append("grid (each data cell shows [row_idx, col_idx]=value):")
    # Markdown header row for readability.
    if headers:
        lines.append("| row | " + " | ".join(headers) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in headers) + " |")
    for r_idx, row in enumerate(rows):
        cells = list(row or [])
        width = max(len(headers), len(cells))
        cells = (cells + [""] * width)[:width]
        rendered = []
        for c_idx, cell in enumerate(cells):
            rendered.append(f"[{r_idx},{c_idx}]={_cell(cell)}")
        lines.append(f"| {r_idx} | " + " | ".join(rendered) + " |")
    return "\n".join(lines)


# ── Legacy aliases (tests / older callers) ───────────────────────────────────


def table_summary_text(
    table: Union[Dict[str, Any], TablePayload],
    *,
    filename: str = "",
    section: str = "",
    subsection: str = "",
    include_llm: bool = False,
) -> str:
    """Backward-compatible alias → ``table_card_text`` (always includes LLM)."""
    _ = include_llm  # card always embeds understanding when present
    return table_card_text(
        table, filename=filename, section=section, subsection=subsection
    )


def table_interpretation_text(
    table: Union[Dict[str, Any], TablePayload],
    *,
    filename: str = "",
    section: str = "",
    subsection: str = "",
) -> str:
    """Deprecated separate interpretation embedding — folded into table_card."""
    return table_card_text(
        table, filename=filename, section=section, subsection=subsection
    )


def table_row_text(
    table: Union[Dict[str, Any], TablePayload],
    row_index: int,
    row: Sequence[Any],
    *,
    filename: str = "",
    section: str = "",
) -> str:
    """Backward-compatible single-row window."""
    return table_row_window_text(
        table,
        [int(row_index)],
        [row],
        filename=filename,
        section=section,
    )


def compact_table_prompt_block(
    table: Union[Dict[str, Any], TablePayload],
    *,
    max_rows: int = 20,
    relevant_row_indices: Optional[Sequence[int]] = None,
) -> str:
    """Structured prompt appendix with headers, rows, spans, and warnings."""
    data = _as_dict(table)
    headers = [str(h or "") for h in (data.get("headers") or [])]
    rows = [list(r) for r in (data.get("rows") or [])]
    pages = data.get("pages") or []
    page = data.get("page") or (pages[0] if pages else "?")
    caption = _cell(data.get("caption") or "")
    merged = data.get("merged_cells") or []
    low_conf = data.get("low_confidence_cells") or []
    table_id = str(data.get("table_id") or "")

    page_label = (
        ",".join(str(p) for p in pages) if pages and len(pages) > 1 else str(page)
    )
    lines = [
        f"Structured table (page {page_label}"
        + (f", id={table_id}" if table_id else "")
        + "):"
    ]
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
    if low_conf or data.get("needs_verification"):
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
    llm_block = _understanding_index_text(data)
    if llm_block:
        lines.append("")
        lines.append(llm_block)
    return "\n".join(lines)
