"""HTML exporter with optional merged-cell (colspan/rowspan) support.

Cells carrying ``colspan``/``rowspan`` > 1 are emitted once with the
appropriate span attribute; positions covered by a span are skipped so the
effective column count per row stays correct.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ste.models import ExtractedTable, normalize_ws


def _esc(value: Any) -> str:
    text = normalize_ws(str(value or ""))
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _cell_tag(tag: str, text: str, colspan: int, rowspan: int) -> str:
    attrs = []
    if colspan > 1:
        attrs.append(f'colspan="{colspan}"')
    if rowspan > 1:
        attrs.append(f'rowspan="{rowspan}"')
    attr_str = f" {' '.join(attrs)}" if attrs else ""
    return f"<{tag}{attr_str}>{_esc(text)}</{tag}>"


def table_to_html(table: ExtractedTable) -> str:
    width = max(len(table.headers or []), max((len(r) for r in table.rows or []), default=0), 1)
    headers = list(table.headers or []) + [""] * width
    headers = headers[:width]
    parts = [
        '<table class="ste-data-table" border="1" cellspacing="0" cellpadding="4">',
    ]
    if table.caption:
        parts.append(f"<caption>{_esc(table.caption)}</caption>")
    parts.append("<thead><tr>")
    for h in headers:
        parts.append(f"<th>{_esc(h)}</th>")
    parts.append("</tr></thead><tbody>")
    # Build a set of (row, col) positions covered by a span (non-origin cells
    # of a merged region) so they are not emitted as separate <td> elements.
    covered: set[tuple[int, int]] = set()
    span_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for cell in table.cells or []:
        key = (cell.row, cell.col)
        span_lookup[key] = (cell.colspan, cell.rowspan)
        if cell.colspan > 1 or cell.rowspan > 1:
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    if (dr, dc) != (0, 0):
                        covered.add((cell.row + dr, cell.col + dc))
    for row_idx, row in enumerate(table.rows or []):
        cells = list(row) + [""] * width
        cells = cells[:width]
        parts.append("<tr>")
        for col_idx, cell_text in enumerate(cells):
            if (row_idx, col_idx) in covered:
                continue
            colspan, rowspan = span_lookup.get((row_idx, col_idx), (1, 1))
            parts.append(_cell_tag("td", cell_text, colspan, rowspan))
        parts.append("</tr>")
    parts.append("</tbody></table>")
    html = "".join(parts)
    return html


def write_html(path: Path, table: ExtractedTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = table_to_html(table)
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(table.table_id)}</title>"
        "<style>table{border-collapse:collapse;font-family:sans-serif;font-size:12px}"
        "th,td{border:1px solid #333;padding:4px 6px;white-space:nowrap}"
        "th{background:#eee}</style></head><body>"
        f"{body}</body></html>"
    )
    path.write_text(doc, encoding="utf-8")
