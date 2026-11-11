"""Generic table normalization: merges, header hierarchy, label inheritance.

These algorithms are structural only. They never know what a column or row
means; they resolve geometry (spans), repeated/inherited labels, stacked
multi-line cells and multi-value cells so that any table from any domain becomes
a rectangular, verifiable grid.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.rag.ingestion.pdf_layout.types import TableData, normalize_ws
from app.rag.tables.schema import TableCell, TableRecord, make_table_id

_RANGE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:[-\u2212\u2013\u2014]|to)\s*[>≥]?\s*\d+(?:\.\d+)?",
    re.I,
)
_LINE_SPLIT_RE = re.compile(r"[\r\n]+")


def _norm(value: Any) -> str:
    return normalize_ws(str(value or ""))


def _raw(value: Any) -> str:
    """Preserve line breaks (needed for stacked labels/header splitting)."""
    text = str(value or "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def rectangularize(rows: Sequence[Sequence[Any]], *, width: Optional[int] = None) -> List[List[str]]:
    values = [[_raw(cell) for cell in (row or [])] for row in (rows or [])]
    width = width or max((len(row) for row in values), default=0)
    return [(row + [""] * width)[:width] for row in values]


def _contains_range(text: str) -> bool:
    return bool(_RANGE_RE.search(_norm(text)))


def _is_pure_numeric(cell: str) -> bool:
    text = _norm(cell)
    if not text:
        return False
    return bool(
        re.fullmatch(
            r"(?:[$€£₹]?\s*)?[-+]?\d+(?:[.,]\d+)*(?:\s*%|\s*[A-Za-z]{1,4})?",
            text,
        )
    )


def _split_value_tokens(cell: str) -> List[str]:
    text = _norm(cell)
    if not text:
        return []
    if re.search(r"\s[/|]\s", text):
        parts = [_norm(p) for p in re.split(r"\s*[/|]\s*", text)]
        if len(parts) >= 2 and all(parts):
            return parts
    tokens = re.findall(r"\d+(?:\.\d+)?(?:\s*[A-Za-z%]{1,6})?", text)
    if len(tokens) >= 2:
        remainder = re.sub(r"\d+(?:\.\d+)?(?:\s*[A-Za-z%]{1,6})?", "", text)
        if len(_norm(remainder)) <= 2:
            return [_norm(token) for token in tokens]
    return [text]


def _is_value_cell(cell: str) -> bool:
    """A measured value: one number, or several numbers collapsed together.

    Range labels (``10-20``, ``>70-90 kg``) are labels, not measurements.
    """
    text = _norm(cell)
    if not text or _contains_range(text):
        return False
    if _is_pure_numeric(text):
        return True
    tokens = _split_value_tokens(text)
    return len(tokens) >= 2 and all(_is_pure_numeric(token) for token in tokens)


def _numeric_ratio(row: Sequence[str]) -> float:
    filled = [cell for cell in row if _norm(cell)]
    if not filled:
        return 0.0
    return sum(1 for cell in filled if _is_value_cell(cell)) / len(filled)


def drop_empty_columns(
    grid: Sequence[Sequence[Any]],
    spans: Optional[Sequence[Dict[str, Any]]] = None,
    cells: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[List[str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove columns with no content in any row and remap geometry."""
    values = rectangularize(grid)
    if not values:
        return values, list(spans or []), list(cells or [])
    width = len(values[0])
    keep = [col for col in range(width) if any(_norm(row[col]) for row in values)]
    if len(keep) == width or len(keep) < 2:
        return values, list(spans or []), list(cells or [])

    col_map = {old: new for new, old in enumerate(keep)}
    new_grid = [[row[col] for col in keep] for row in values]

    new_spans: List[Dict[str, Any]] = []
    for span in spans or []:
        try:
            row = int(span.get("row", 0))
            col = int(span.get("col", 0))
            row_span = max(1, int(span.get("row_span", 1)))
            col_span = max(1, int(span.get("col_span", 1)))
        except (TypeError, ValueError):
            continue
        if col not in col_map:
            continue
        kept = [c for c in range(col, col + col_span) if c in col_map]
        if not kept:
            continue
        new_spans.append(
            {
                "row": row,
                "col": col_map[kept[0]],
                "row_span": row_span,
                "col_span": len(kept),
            }
        )

    new_cells: List[Dict[str, Any]] = []
    for cell in cells or []:
        try:
            col = int(cell.get("col", 0))
        except (TypeError, ValueError):
            continue
        if col not in col_map:
            continue
        item = dict(cell)
        item["col"] = col_map[col]
        merged = cell.get("merged_into")
        if isinstance(merged, (list, tuple)) and len(merged) == 2:
            if int(merged[1]) not in col_map:
                continue
            item["merged_into"] = [int(merged[0]), col_map[int(merged[1])]]
        new_cells.append(item)

    return new_grid, new_spans, new_cells


def _left_align_span(row: List[str], start: int, end: int) -> None:
    """Left-align non-empty values inside ``row[start:end]`` when leading cells are blank.

    Parsers often right-pack an incomplete value row into the last column of a
    header group that was split from one stacked cell. Leading empties + a
    trailing value become left-aligned; rows that already start with a value are
    left alone so sparse middle blanks stay intact.
    """
    if end < start or start < 0 or end >= len(row):
        return
    cells = row[start : end + 1]
    values = [cell for cell in cells if _norm(cell)]
    if not values or len(values) == len(cells):
        return
    if _norm(cells[0]):
        return
    for offset, col in enumerate(range(start, end + 1)):
        row[col] = values[offset] if offset < len(values) else ""


def expand_multiline_cells(
    grid: Sequence[Sequence[Any]],
    header_count: int,
) -> List[List[str]]:
    """Spread line-stacked cell content into the cells it visually spans.

    Extraction frequently puts ``150 / 150 / 150 / 300`` in one cell while the
    four value columns sit empty beside it, or stacks several row labels in one
    first-column cell. Expansion is limited to empty sibling cells so no real
    value is overwritten.
    """
    rows = rectangularize(grid)
    if not rows:
        return rows
    width = len(rows[0])
    out = [list(row) for row in rows]
    # Columns created by splitting a stacked header cell. Body rows under these
    # spans are left-aligned after expansion (see below).
    stacked_header_spans: List[Tuple[int, int]] = []
    for r, row in enumerate(out):
        for c in range(width):
            parts = [_norm(part) for part in _LINE_SPLIT_RE.split(row[c] or "") if _norm(part)]
            if len(parts) < 2:
                row[c] = _norm(row[c])
                continue
            if c == 0 and r >= header_count:
                # Vertical label stack: distribute only when the following rows
                # (including this one) are otherwise value rows.
                window = out[r : r + len(parts)]
                value_rows = sum(
                    1
                    for candidate in window
                    if any(_norm(candidate[k]) for k in range(1, width))
                )
                if len(window) == len(parts) and value_rows == len(parts):
                    for offset, part in enumerate(parts):
                        out[r + offset][0] = part
                    continue
            if (
                c > 0
                and r >= header_count
                and len(parts) >= 3
                and r + len(parts) <= len(out)
                and all(_looks_like_phrase_fragment(part) for part in parts)
                and all(
                    not _norm(out[k][c])
                    for k in range(r + 1, r + len(parts))
                )
            ):
                # A parser can place a vertically fragmented phrase block
                # ("Every / 4 / weeks / ...") in one cell; spread it back down
                # its rows so phrase coalescing can rebuild each block.
                for offset, part in enumerate(parts):
                    out[r + offset][c] = part
                continue
            siblings = list(range(c + 1, min(width, c + len(parts))))
            if siblings and all(not _norm(row[k]) for k in siblings):
                for offset, part in enumerate(parts):
                    if c + offset < width:
                        row[c + offset] = part
                if r < header_count:
                    stacked_header_spans.append((c, c + len(parts) - 1))
                continue
            # Some parsers right-align a merged multi-column cell into the last
            # column; spread it left into the empty siblings in that case.
            left = list(range(max(0, c - len(parts) + 1), c))
            if left and len(left) == len(parts) - 1 and all(
                not _norm(row[k]) for k in left
            ):
                for offset, part in enumerate(parts):
                    row[left[0] + offset] = part
                if r < header_count:
                    stacked_header_spans.append((left[0], c))
            else:
                # Not a distributable stack: keep it as one readable label.
                row[c] = " ".join(parts)

    if stacked_header_spans:
        # Deduplicate overlapping spans by keeping the widest coverage per start.
        spans = sorted({(int(a), int(b)) for a, b in stacked_header_spans if b > a})
        for r in range(header_count, len(out)):
            for start, end in spans:
                if end < width:
                    _left_align_span(out[r], start, end)
    return out


def detect_header_row_count(
    grid: Sequence[Sequence[str]],
    *,
    hint: Optional[int] = None,
) -> int:
    """Header rows are everything before the first row with a measured value.

    Ranges, units and text labels do not count as values, so multi-row headers
    (group + leaf labels + units) stay in the header band. ``hint`` from the
    extractor is used only when the grid has no measurable values at all.
    """
    rows = rectangularize(grid)
    if len(rows) <= 1:
        return 1
    for index, row in enumerate(rows):
        candidates = row[1:] if len(row) > 1 else row
        if any(_is_value_cell(cell) for cell in candidates):
            return max(1, min(index, len(rows) - 1))
    if hint is not None and hint >= 1:
        return max(1, min(int(hint), len(rows) - 1))
    return 1


def resolve_merged_cells(
    grid: Sequence[Sequence[str]],
    spans: Sequence[Dict[str, Any]],
) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    """Fill covered coordinates with their anchor cell's value."""
    rows = rectangularize(grid)
    if not rows:
        return rows, []
    height, width = len(rows), len(rows[0])
    normalized: List[Dict[str, Any]] = []
    for span in spans or []:
        try:
            row = int(span.get("row", 0))
            col = int(span.get("col", 0))
            row_span = max(1, int(span.get("row_span", 1)))
            col_span = max(1, int(span.get("col_span", 1)))
        except (TypeError, ValueError):
            continue
        if row < 0 or col < 0 or row >= height or col >= width:
            continue
        anchor = _norm(rows[row][col])
        if not anchor:
            anchor = next(
                (
                    _norm(rows[r][c])
                    for r in range(row, min(height, row + row_span))
                    for c in range(col, min(width, col + col_span))
                    if _norm(rows[r][c])
                ),
                "",
            )
        normalized.append(
            {"row": row, "col": col, "row_span": row_span, "col_span": col_span}
        )
        for r in range(row, min(height, row + row_span)):
            for c in range(col, min(width, col + col_span)):
                if (r, c) == (row, col):
                    continue
                if not _norm(rows[r][c]):
                    rows[r][c] = anchor
    return rows, normalized


def merge_header_rows(header_rows: Sequence[Sequence[str]]) -> List[str]:
    """Collapse a multi-row header into one resolved label per column."""
    rows = rectangularize(header_rows)
    if not rows:
        return []
    width = len(rows[0])
    labels: List[str] = []
    for col in range(width):
        parts: List[str] = []
        for row in rows:
            for part in _LINE_SPLIT_RE.split(row[col] or ""):
                value = _norm(part)
                if value and (not parts or parts[-1] != value):
                    parts.append(value)
        labels.append(" / ".join(parts) or f"Column {col + 1}")
    return labels


def infer_label_columns(
    rows: Sequence[Sequence[str]],
    *,
    max_label_columns: int = 3,
) -> List[int]:
    """Leading columns that carry row labels rather than measured values."""
    values = rectangularize(rows)
    if not values:
        return [0]
    width = len(values[0])
    columns = [0]
    for col in range(1, min(max_label_columns, width)):
        filled = [_norm(row[col]) for row in values if _norm(row[col])]
        if not filled:
            break
        value_ratio = sum(1 for cell in filled if _is_value_cell(cell)) / len(filled)
        range_ratio = sum(1 for cell in filled if _contains_range(cell)) / len(filled)
        lengths = sorted(len(cell) for cell in filled)
        median_len = lengths[len(lengths) // 2]
        alpha_only = all(
            re.search(r"[A-Za-z]", cell) and not re.search(r"\d", cell)
            for cell in filled
        )
        # Only short, label-like columns become extra row-label columns. A
        # long phrase cell (e.g. a schedule description) must stay a value.
        if value_ratio <= 0.34 and max(len(cell) for cell in filled) <= 24 and (
            median_len >= 6 or range_ratio >= 0.5 or alpha_only
        ):
            columns.append(col)
        else:
            break
    return columns


def forward_fill_hierarchical_labels(
    rows: Sequence[Sequence[str]],
    label_columns: Sequence[int],
) -> List[List[str]]:
    """Fill inherited labels in a hierarchical (rowspan/grouped) label column."""
    values = rectangularize(rows)
    last: Dict[int, str] = {}
    for row in values:
        if not any(_norm(cell) for cell in row):
            continue
        for col in label_columns:
            if col >= len(row):
                continue
            current = _norm(row[col])
            if current:
                last[col] = current
            elif last.get(col):
                row[col] = last[col]
    return values


def _looks_like_phrase_fragment(cell: str) -> bool:
    text = _norm(cell)
    if not text or len(text) > 24 or _contains_range(text):
        return False
    return len(text.split()) <= 3


def _column_is_phrase_column(rows: Sequence[Sequence[str]], col: int) -> bool:
    values = [_norm(row[col]) for row in rows if col < len(row) and _norm(row[col])]
    if not values:
        return False
    if any(re.search(r"\d{2,}", value) for value in values):
        return False
    if any(_contains_range(value) for value in values):
        return False
    return any(re.search(r"[A-Za-z]", value) for value in values)


def coalesce_vertical_phrases(
    rows: Sequence[Sequence[str]],
    label_columns: Sequence[int],
) -> List[List[str]]:
    """Join vertically fragmented phrases (``Every`` / ``4`` / ``weeks``).

    Applies only to non-label columns whose contents are short fragments with
    no multi-digit measurements, so numeric value columns are never merged.
    """
    values = rectangularize(rows)
    if not values:
        return values
    width = len(values[0])
    out = [[_norm(cell) for cell in row] for row in values]
    for col in range(width):
        if col in label_columns or not _column_is_phrase_column(values, col):
            continue
        index = 0
        while index < len(out):
            fragments: List[str] = []
            cursor = index
            while (
                cursor < len(out)
                and _looks_like_phrase_fragment(out[cursor][col])
                and len(fragments) < 6
            ):
                fragments.append(_norm(out[cursor][col]))
                cursor += 1
            has_alpha = any(re.search(r"[A-Za-z]", part) for part in fragments)
            has_digit = any(re.search(r"\d", part) for part in fragments)
            if len(fragments) >= 3 and has_alpha and has_digit:
                # A repeated head token (e.g. "Every 4 weeks / Every 2 weeks")
                # starts a new block; segment before assigning.
                groups: List[List[str]] = [[fragments[0]]]
                for fragment in fragments[1:]:
                    if (
                        fragment.lower() == fragments[0].lower()
                        and len(groups[-1]) >= 2
                    ):
                        groups.append([fragment])
                    else:
                        groups[-1].append(fragment)
                start = index
                for group in groups:
                    phrase = " ".join(group)
                    for offset in range(len(group)):
                        if start + offset < len(out):
                            out[start + offset][col] = phrase
                    start += len(group)
                # Inherit the last phrase for trailing blank rows of the block.
                while start < len(out) and not out[start][col]:
                    out[start][col] = " ".join(groups[-1])
                    start += 1
                index = max(cursor, start)
            else:
                index += 1
    # A column that repeats one or two phrases across a block (a schedule) is
    # inherited by its blank rows; measured-value columns are never touched.
    for col in range(width):
        if col in label_columns or not _column_is_phrase_column(values, col):
            continue
        filled = [_norm(row[col]) for row in out if _norm(row[col])]
        if not filled or len(set(filled)) > 2:
            continue
        if not any(not _norm(row[col]) for row in out):
            continue
        last = ""
        for row in out:
            if _norm(row[col]):
                last = _norm(row[col])
            elif last:
                row[col] = last
    return out


def merge_continuation_rows(
    rows: Sequence[Sequence[str]],
    label_columns: Sequence[int],
) -> List[List[str]]:
    """Fold a label-only continuation row into the row it continues.

    Some parsers emit a second row for a merged label cell that carries only
    the remaining label text (for example a schedule that applies to the
    previous row); the row has no measured values and repeats the primary
    label. Its text is merged into the preceding row's blanks.
    """
    values = rectangularize(rows)
    if not values:
        return values
    width = len(values[0])
    primary = label_columns[0] if label_columns else 0
    out: List[List[str]] = []
    for row in values:
        normalized = [_norm(cell) for cell in row]
        if not any(normalized):
            continue
        has_measurement = any(
            _is_value_cell(normalized[col])
            for col in range(width)
            if col != primary
        )
        if (
            out
            and normalized[primary]
            and normalized[primary] == out[-1][primary]
            and not has_measurement
        ):
            for col in range(width):
                if normalized[col] and not out[-1][col]:
                    out[-1][col] = normalized[col]
            continue
        out.append(normalized)
    return out


def _header_group(label: str) -> str:
    return _norm(label).split(" / ")[0].strip().lower()


def distribute_multi_value_cells(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    label_columns: Sequence[int],
) -> List[List[str]]:
    """Spread a collapsed multi-value cell across sibling columns."""
    values = rectangularize(rows)
    if not values:
        return values
    width = len(values[0])
    groups: Dict[str, List[int]] = {}
    for col in range(width):
        groups.setdefault(_header_group(headers[col] if col < len(headers) else ""), []).append(col)
    for row in values:
        for group in groups.values():
            if len(group) < 2:
                continue
            anchor = group[0]
            if anchor in label_columns:
                continue
            tokens = _split_value_tokens(row[anchor])
            if len(tokens) != len(group):
                continue
            sibling_values = [_norm(row[col]) for col in group[1:]]
            if any(sibling_values):
                continue
            for col, token in zip(group, tokens):
                row[col] = token
    return values


def linearize_facts(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    row_labels: Optional[Sequence[str]] = None,
    label_columns: Sequence[int] = (0,),
    caption: str = "",
    max_facts: int = 200,
) -> List[str]:
    """Row x column x value facts for lexical/vector retrieval."""
    values = rectangularize(rows)
    if not values:
        return []
    width = len(values[0])
    labels = list(headers) + [f"Column {i + 1}" for i in range(len(headers), width)]
    facts: List[str] = []
    prefix = _norm(caption)
    for index, row in enumerate(values):
        label = _norm(row_labels[index]) if row_labels and index < len(row_labels) else ""
        if not label:
            label = " ".join(
                _norm(row[col]) for col in label_columns if col < width and _norm(row[col])
            )
        for col in range(width):
            if col in label_columns:
                continue
            value = _norm(row[col])
            if not value:
                continue
            parts = [part for part in (label, labels[col], value) if part]
            fact = " | ".join(parts)
            if prefix and prefix.lower() not in fact.lower():
                fact = f"{prefix}: {fact}"
            facts.append(fact)
            if len(facts) >= max_facts:
                return facts
    return facts


def _row_labels(rows: Sequence[Sequence[str]], label_columns: Sequence[int]) -> List[str]:
    labels: List[str] = []
    for row in rows:
        parts = [
            _norm(row[col])
            for col in label_columns
            if col < len(row) and _norm(row[col])
        ]
        labels.append(" ".join(parts) if parts else next((_norm(c) for c in row if _norm(c)), ""))
    return labels


def normalize_grid(
    headers: Sequence[Any],
    rows: Sequence[Sequence[Any]],
    *,
    grid: Optional[Sequence[Sequence[Any]]] = None,
    header_row_count: int = 1,
    spans: Optional[Sequence[Dict[str, Any]]] = None,
    caption: str = "",
) -> Dict[str, Any]:
    """Normalize an arbitrary grid and return the resolved pieces."""
    if grid:
        raw = [list(row) for row in grid]
    else:
        width = max([len(headers or [])] + [len(row) for row in rows or []] or [0])
        head = list(headers or [])
        head = (head + [""] * width)[:width] if width else []
        raw = ([head] if head else [[""] * width]) + [
            list(row) for row in (rows or [])
        ]

    raw, spans_out, _ = drop_empty_columns(raw, spans or [], [])
    hint = max(1, header_row_count or 1)
    count = detect_header_row_count(raw, hint=hint)
    raw = expand_multiline_cells(raw, count)
    resolved, normalized_spans = resolve_merged_cells(raw, spans_out)
    count = detect_header_row_count(resolved, hint=hint)
    header_rows = resolved[:count]
    body = resolved[count:]
    headers_out = merge_header_rows(header_rows)

    width = max(len(headers_out), max((len(row) for row in body), default=0))
    headers_out = (
        headers_out
        + [f"Column {i + 1}" for i in range(len(headers_out), width)]
    )[:width]
    body = rectangularize(body, width=width)

    label_columns = infer_label_columns(body)
    body = forward_fill_hierarchical_labels(body, label_columns)
    body = coalesce_vertical_phrases(body, label_columns)
    body = merge_continuation_rows(body, label_columns)
    body = distribute_multi_value_cells(headers_out, body, label_columns)
    body = [[_norm(cell) for cell in row] for row in body]
    headers_out = [_norm(header) for header in headers_out]
    row_labels = _row_labels(body, label_columns)
    facts = linearize_facts(
        headers_out,
        body,
        row_labels=row_labels,
        label_columns=label_columns,
        caption=caption,
    )
    return {
        "headers": headers_out,
        "rows": body,
        "header_row_count": count,
        "spans": normalized_spans,
        "label_columns": label_columns,
        "row_labels": row_labels,
        "facts": facts,
    }


def normalize_table_data(table: TableData, *, document_id: str = "") -> TableData:
    """In-place normalization for the layout pipeline's ``TableData``."""
    grid = getattr(table, "grid", None) or None
    normalized = normalize_grid(
        table.headers,
        table.rows,
        grid=grid,
        header_row_count=max(1, getattr(table, "header_row_count", 1) or 1),
        spans=getattr(table, "spans", None) or [],
        caption=table.caption or "",
    )
    table.headers = normalized["headers"]
    table.rows = normalized["rows"]
    table.header_row_count = normalized["header_row_count"]
    table.spans = normalized["spans"]
    table.label_columns = normalized["label_columns"]
    table.row_labels = normalized["row_labels"]
    table.linearized_facts = normalized["facts"]
    if document_id or not table.table_id:
        table.table_id = make_table_id(
            document_id=document_id,
            page=table.page,
            bbox=list(table.bbox.as_tuple()) if table.bbox else [],
            headers=table.headers,
            rows=table.rows,
        )
    return table


def record_from_table_data(
    table: TableData,
    *,
    document_id: str = "",
    filename: str = "",
    source_index: int = 0,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> TableRecord:
    normalize_table_data(table, document_id=document_id)
    grid = getattr(table, "grid", None) or []
    cells = [
        TableCell(
            row=int(cell.get("row", 0)),
            col=int(cell.get("col", 0)),
            text=_norm(cell.get("text")),
            row_span=max(1, int(cell.get("row_span", 1) or 1)),
            col_span=max(1, int(cell.get("col_span", 1) or 1)),
            header=bool(cell.get("header")),
            merged_into=tuple(cell["merged_into"]) if cell.get("merged_into") else None,
            bbox=list(cell.get("bbox") or []) or None,
            confidence=float(cell.get("confidence", 100.0) or 100.0),
        )
        for cell in (getattr(table, "cells", None) or [])
        if isinstance(cell, dict)
    ]
    if not cells and grid:
        for row_index, row in enumerate(grid):
            for col_index, value in enumerate(row):
                if _norm(value):
                    cells.append(
                        TableCell(
                            row=row_index,
                            col=col_index,
                            text=_norm(value),
                            header=row_index < max(1, table.header_row_count),
                        )
                    )
    return TableRecord(
        table_id=table.table_id or make_table_id(
            document_id=document_id,
            page=table.page,
            bbox=list(table.bbox.as_tuple()) if table.bbox else [],
            headers=table.headers,
            rows=table.rows,
        ),
        document_id=str(document_id or ""),
        page=int(table.page or 1),
        bbox=list(table.bbox.as_tuple()) if table.bbox else [],
        page_width=page_width,
        page_height=page_height,
        caption=_norm(table.caption),
        footnotes=[_norm(f) for f in (getattr(table, "footnotes", None) or []) if _norm(f)],
        headers=list(table.headers),
        rows=[list(row) for row in table.rows],
        header_row_count=max(1, table.header_row_count),
        row_labels=list(table.row_labels),
        label_columns=list(table.label_columns),
        cells=cells,
        spans=list(table.spans or []),
        original={
            "grid": grid,
            "markdown": getattr(table, "original_markdown", "") or table.markdown,
        },
        extraction_method=table.extraction_method,
        source_lines=table.source_lines,
        filename=filename,
        source_index=source_index,
    )
