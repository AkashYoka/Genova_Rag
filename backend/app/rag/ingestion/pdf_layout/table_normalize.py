"""Generic matrix/table repair and retrieval-friendly linearization.

Complex PDF tables often use multi-row headers, merged cells, and labels split
across rows (e.g. Every / 4 / weeks). Geometry extraction can smash row labels
into one cell and break row×column relationships. This module:

1. Detects collapsed / low-quality grids.
2. Forward-fills fragmented schedule/interval labels.
3. Promotes range-like header bands.
4. Emits linearized cell facts for retrieval and LLM context.
5. Stitches multipage table continuations.

Domain-agnostic: works for any row-label × column-header matrix
(schedules, rate cards, invoices, lab grids, engineering specs, etc.).
No Camelot / third-party table repair.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.rag.ingestion.pdf_layout.types import normalize_ws

logger = logging.getLogger(__name__)

# Range-like tokens commonly used as row/column headers in dosing tables.
_RANGE_RE = re.compile(
    r"(?:≥\s*)?(?:>\s*)?\d+(?:\.\d+)?\s*[−\-–—to]+\s*(?:>\s*)?\d+(?:\.\d+)?",
    re.I,
)
_FREQ_FRAG_RE = re.compile(
    r"^(every|q\d*|once|twice|\d+|weeks?|days?|months?|hours?|hrs?)$",
    re.I,
)
_COMPLETE_FREQ_RE = re.compile(
    r"every\s+\d+\s+(?:weeks?|days?|months?|hours?|hrs?)",
    re.I,
)
# Numeric value cells (optional short unit suffix) — domain-agnostic.
_NUMERIC_VALUE_RE = re.compile(
    r"^\d+(?:\.\d+)?(?:\s*[a-zA-Z%/μµ²³]{0,6})?$", re.I
)


@dataclass
class NormalizedTable:
    headers: List[str]
    rows: List[List[str]]
    facts: List[str]
    repaired: bool = False
    method: str = "native"
    quality_notes: Tuple[str, ...] = ()


def split_range_labels(cell: str) -> List[str]:
    """Extract discrete range labels from a possibly concatenated cell."""
    text = normalize_ws(cell or "")
    if not text:
        return []
    matches = [normalize_ws(m.group(0)) for m in _RANGE_RE.finditer(text)]
    return matches


def looks_collapsed_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> bool:
    """True when row labels appear smashed or largely missing."""
    if not rows:
        return False
    smashed = 0
    empty_label = 0
    numeric_rows = 0
    for row in rows:
        if not row:
            continue
        label = normalize_ws(row[0] if row else "")
        ranges = split_range_labels(label)
        if len(ranges) >= 3:
            smashed += 1
        if not label:
            empty_label += 1
        rest = " ".join(normalize_ws(c) for c in row[1:])
        if any(_NUMERIC_VALUE_RE.match(normalize_ws(c)) for c in row[1:]):
            numeric_rows += 1
        elif re.search(r"\b\d{2,4}\b", rest):
            numeric_rows += 1
    if smashed:
        return True
    if len(rows) >= 4 and empty_label >= max(2, len(rows) // 2) and numeric_rows >= 3:
        return True
    # A single header *cell* that itself contains many ranges is collapsed;
    # do not flag normal multi-column weight headers (each cell is one range).
    for header_cell in headers or []:
        if len(split_range_labels(header_cell)) >= 3:
            return True
    return False


def coalesce_frequency_fragments(rows: List[List[str]], freq_col: int = 1) -> List[List[str]]:
    """Merge fragmented frequency cells (Every / 4 / weeks) into full phrases."""
    if not rows or freq_col < 0:
        return rows
    width = max(len(r) for r in rows)
    if freq_col >= width:
        return rows

    out = [list(r) + [""] * (width - len(r)) for r in rows]
    i = 0
    while i < len(out):
        frag = normalize_ws(out[i][freq_col])
        if not frag or not _FREQ_FRAG_RE.match(frag):
            i += 1
            continue
        frags = [frag]
        j = i + 1
        while j < len(out):
            nxt = normalize_ws(out[j][freq_col])
            if not nxt:
                j += 1
                continue
            if nxt.lower() == "every" and frags:
                break
            if _FREQ_FRAG_RE.match(nxt):
                frags.append(nxt)
                j += 1
                phrase = " ".join(frags)
                if _COMPLETE_FREQ_RE.search(phrase):
                    break
                continue
            break
        phrase = " ".join(frags)
        if not _COMPLETE_FREQ_RE.search(phrase) and len(frags) >= 2:
            # Still assign best-effort phrase (e.g. "Every 2 weeks" from parts).
            phrase = " ".join(frags)
        for k in range(i, max(j, i + 1)):
            out[k][freq_col] = phrase
        i = max(j, i + 1)
    # Forward-fill empty frequency cells from previous non-empty.
    last = ""
    for row in out:
        cur = normalize_ws(row[freq_col])
        if cur:
            last = cur
        elif last:
            row[freq_col] = last
    return out


def expand_newline_cells(rows: List[List[str]]) -> List[List[str]]:
    """Split cells that contain newline-stacked values into separate columns when safe.

    Some extractors merge adjacent range columns into one cell with newlines.
    When every stacked cell in a row has the same arity, expand columns.
    """
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [list(r) + [""] * (width - len(r)) for r in rows]

    # Detect columns that consistently contain newline stacks of the same size.
    expand_plan: Dict[int, int] = {}
    for c in range(width):
        arities = []
        for row in padded:
            cell = row[c] or ""
            parts = [normalize_ws(p) for p in re.split(r"[\n\r]+", cell) if normalize_ws(p)]
            if len(parts) >= 2:
                arities.append(len(parts))
        if arities and min(arities) == max(arities) and arities[0] >= 2:
            # Only expand if a majority of non-empty cells are stacked.
            non_empty = sum(1 for row in padded if normalize_ws(row[c]))
            stacked = sum(
                1
                for row in padded
                if len([p for p in re.split(r"[\n\r]+", row[c] or "") if normalize_ws(p)]) >= 2
            )
            if non_empty and stacked >= max(1, non_empty // 2):
                expand_plan[c] = arities[0]

    if not expand_plan:
        # Still flatten newlines to spaces for readability.
        return [
            [normalize_ws((cell or "").replace("\n", " ")) for cell in row]
            for row in padded
        ]

    new_rows: List[List[str]] = []
    for row in padded:
        new_row: List[str] = []
        for c, cell in enumerate(row):
            if c in expand_plan:
                parts = [normalize_ws(p) for p in re.split(r"[\n\r]+", cell or "") if normalize_ws(p)]
                arity = expand_plan[c]
                if not parts:
                    new_row.extend([""] * arity)
                elif len(parts) == arity:
                    new_row.extend(parts)
                elif len(parts) == 1:
                    # Single value for a multi-width column: put in first slot.
                    new_row.append(parts[0])
                    new_row.extend([""] * (arity - 1))
                else:
                    # Pad / trim.
                    parts = (parts + [""] * arity)[:arity]
                    new_row.extend(parts)
            else:
                new_row.append(normalize_ws((cell or "").replace("\n", " ")))
        new_rows.append(new_row)
    return new_rows


def _guess_freq_col(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> int:
    for i, h in enumerate(headers or []):
        if re.search(r"freq|interval|schedule", h or "", re.I):
            return i
    # Heuristic: column dominated by frequency fragments.
    if not rows:
        return 1
    width = max(len(r) for r in rows)
    best_i, best_score = 1 if width > 1 else 0, -1
    for c in range(min(width, 3)):
        score = 0
        for row in rows:
            if c < len(row) and _FREQ_FRAG_RE.match(normalize_ws(row[c])):
                score += 1
            if c < len(row) and _COMPLETE_FREQ_RE.search(normalize_ws(row[c])):
                score += 2
        if score > best_score:
            best_score = score
            best_i = c
    return best_i


def _header_labels(headers: Sequence[str], width: int) -> List[str]:
    labels = [normalize_ws(h) for h in (list(headers) + [""] * width)[:width]]
    # Fill empty headers with Column N for stable facts.
    return [h if h else f"Column {i+1}" for i, h in enumerate(labels)]


def linearize_table_facts(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    caption: str = "",
    max_facts: int = 80,
) -> List[str]:
    """Emit retrieval-friendly facts: row_label | col_header | value [| frequency]."""
    if not rows:
        return []
    width = max([len(headers or [])] + [len(r) for r in rows] or [0])
    if width < 2:
        return []
    col_headers = _header_labels(headers, width)
    freq_col = _guess_freq_col(headers, rows)
    facts: List[str] = []
    prefix = normalize_ws(caption)
    for row in rows:
        cells = list(row) + [""] * (width - len(row))
        row_label = normalize_ws(cells[0])
        freq = normalize_ws(cells[freq_col]) if 0 <= freq_col < width else ""
        if freq_col == 0:
            freq = ""
        for c in range(1, width):
            if c == freq_col:
                continue
            value = normalize_ws(cells[c])
            if not value:
                continue
            # Skip non-value cells (header leftovers, insufficient-data spans handled as values).
            parts = [p for p in (row_label, col_headers[c], value) if p]
            if freq and c != freq_col:
                parts.append(freq)
            fact = " | ".join(parts)
            if prefix and prefix not in fact:
                fact = f"{prefix}: {fact}"
            facts.append(fact)
            if len(facts) >= max_facts:
                return facts
    return facts


def normalize_table_grid(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    caption: str = "",
    method: str = "native",
) -> NormalizedTable:
    """Normalize a grid: expand newlines, coalesce frequencies, emit facts."""
    notes: List[str] = []
    hdrs = [normalize_ws(h) for h in headers]
    body = [[normalize_ws(c) for c in row] for row in rows]
    collapsed = looks_collapsed_table(hdrs, body)
    if collapsed:
        notes.append("collapsed_labels")

    body = expand_newline_cells(body)
    freq_col = _guess_freq_col(hdrs, body)
    body = coalesce_frequency_fragments(body, freq_col=freq_col)

    # If first column still has smashed ranges, try to explode into rows when
    # the number of ranges matches the number of subsequent numeric matrix rows.
    if body and len(split_range_labels(body[0][0] if body[0] else "")) >= 3:
        notes.append("attempted_label_explosion")
        body = _explode_smashed_label_row(body)

    facts = linearize_table_facts(hdrs, body, caption=caption)
    if not facts and caption:
        notes.append("no_facts")
    return NormalizedTable(
        headers=hdrs,
        rows=body,
        facts=facts,
        repaired=False,
        method=method,
        quality_notes=tuple(notes),
    )


def _explode_smashed_label_row(rows: List[List[str]]) -> List[List[str]]:
    """If row0 col0 has N ranges and there are N numeric rows below, zip labels."""
    if not rows:
        return rows
    labels = split_range_labels(rows[0][0] if rows[0] else "")
    if len(labels) < 3:
        return rows
    # Candidate numeric rows: rows after the smashed one with mostly numeric cells.
    candidates: List[List[str]] = []
    for row in rows[1:]:
        numeric = sum(1 for c in row[1:] if re.search(r"\d", c or ""))
        if numeric >= 1:
            candidates.append(row)
    if len(candidates) != len(labels):
        return rows
    rebuilt: List[List[str]] = []
    for label, row in zip(labels, candidates):
        new_row = list(row)
        new_row[0] = label
        rebuilt.append(new_row)
    return rebuilt


def render_table_with_facts(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    caption: str = "",
    section: str = "",
    subsection: str = "",
    include_markdown: bool = True,
) -> str:
    """Build chunk text: section context + markdown + linearized facts."""
    normalized = normalize_table_grid(headers, rows, caption=caption)
    parts: List[str] = []
    for meta in (section, subsection, caption):
        if meta and normalize_ws(meta):
            parts.append(normalize_ws(meta))
    if include_markdown:
        width = max([len(normalized.headers)] + [len(r) for r in normalized.rows] or [0])
        if width:
            hdrs = (normalized.headers + [""] * width)[:width]
            md = [
                "| " + " | ".join(h.replace("|", "\\|") for h in hdrs) + " |",
                "| " + " | ".join("---" for _ in range(width)) + " |",
            ]
            for row in normalized.rows:
                cells = (list(row) + [""] * width)[:width]
                md.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
            parts.append("\n".join(md))
    if normalized.facts:
        parts.append("Table facts:\n" + "\n".join(f"- {fact}" for fact in normalized.facts))
    return "\n\n".join(p for p in parts if p)


def _promote_range_headers(raw_rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """Prefer a row rich in range/unit labels over a sparse title row."""
    if len(raw_rows) < 2:
        return raw_rows[0], raw_rows[1:] if len(raw_rows) > 1 else []

    def range_header_score(row: Sequence[str]) -> int:
        score = 0
        for cell in row:
            text = normalize_ws(cell)
            # Any short unit token (kg, %, USD, MPa, …) strengthens header candidacy.
            if re.search(r"(?i)\b[a-z]{1,6}\b", text) and re.search(r"\d", text):
                score += 1
            score += len(split_range_labels(text))
        return score

    best_i = 0
    best_score = range_header_score(raw_rows[0])
    for i, row in enumerate(raw_rows[:4]):
        score = range_header_score(row)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i == 0 or best_score < 2:
        return raw_rows[0], raw_rows[1:]
    headers = list(raw_rows[best_i])
    for prior in raw_rows[:best_i]:
        for c, cell in enumerate(prior):
            if c >= len(headers):
                break
            if not normalize_ws(headers[c]) and normalize_ws(cell):
                headers[c] = normalize_ws(cell)
    if not normalize_ws(headers[0]):
        headers[0] = "Row label"
    body = raw_rows[best_i + 1 :]
    # Drop sparse unit/banner spacer rows (mostly empty + a unit/label token).
    body = [
        row
        for row in body
        if not (
            sum(1 for c in row if normalize_ws(c)) <= 2
            and any(
                re.search(r"(?i)^(?:unit|value|amount|total|rate|%|\(\s*[a-z]{1,6}\s*\))$", c or "")
                for c in row
            )
        )
    ]
    return headers, body


def _norm_header_key(headers: Sequence[str]) -> Tuple[str, ...]:
    return tuple(normalize_ws(str(h or "")).lower() for h in headers)


def _cols_match(a: Sequence[Sequence[str]], b: Sequence[Sequence[str]]) -> bool:
    def width(grid: Sequence[Sequence[str]]) -> int:
        if not grid:
            return 0
        return max(len(row) for row in grid)

    # Compare header widths when present, else data-row widths.
    wa = width(a)
    wb = width(b)
    return wa > 0 and wa == wb


def _headers_repeated(prev_headers: Sequence[str], next_headers: Sequence[str]) -> bool:
    if not prev_headers or not next_headers:
        return False
    return _norm_header_key(prev_headers) == _norm_header_key(next_headers)


def _looks_continuation_bbox(
    prev: Any,
    nxt: Any,
    *,
    page_height_prev: float = 792.0,
    page_height_next: float = 792.0,
    bottom_frac: float = 0.28,
    top_frac: float = 0.35,
) -> bool:
    """True when prev sits near page bottom and next near page top."""
    pb = getattr(prev, "bbox", None)
    nb = getattr(nxt, "bbox", None)
    if pb is None or nb is None:
        # Missing geometry → allow continuation when other signals match.
        return True
    try:
        prev_bottom = float(getattr(pb, "y1", 0) or 0)
        next_top = float(getattr(nb, "y0", 0) or 0)
        ph = float(page_height_prev or 792.0)
        nh = float(page_height_next or 792.0)
        near_bottom = prev_bottom >= ph * (1.0 - bottom_frac)
        near_top = next_top <= nh * top_frac
        return near_bottom and near_top
    except Exception:
        return True


def _strip_repeated_header_row(
    prev_headers: Sequence[str],
    next_headers: Sequence[str],
    next_rows: List[List[str]],
) -> Tuple[List[str], List[List[str]], bool]:
    """If next page repeats leaf headers, drop the duplicate header row."""
    headers = list(next_headers or [])
    rows = [list(r) for r in next_rows]
    if _headers_repeated(prev_headers, headers):
        return list(prev_headers), rows, True
    # Continuation with no header row (headers empty / placeholder).
    if not any(normalize_ws(h) for h in headers):
        return list(prev_headers), rows, True
    # First data row equals previous headers → treat as repeated header.
    if rows and _headers_repeated(prev_headers, [str(c or "") for c in rows[0]]):
        return list(prev_headers), rows[1:], True
    return headers, rows, False


def stitch_multipage_tables(
    pages: Sequence[Any],
) -> List[Any]:
    """Merge consecutive cross-page table continuations in-place on pages.

    Invoked after Textract extraction + header reconstruction / repair.
    Two consecutive tables on page N and N+1 are stitched when:
      (a) normalized column counts match
      (b) page N+1 repeats identical leaf headers (strip them) OR has no header
      (c) vertical bboxes indicate page-bottom → page-top continuation

    Mutates ``page.tables`` lists and sets ``table.pages`` on survivors.
    Returns the flat list of post-stitch tables for convenience.
    """
    page_list = list(pages or [])
    if not page_list:
        return []

    # Ensure every table has a pages list seeded from its page number.
    for page in page_list:
        page_h = float((getattr(page, "metrics", None) or {}).get("page_height") or 792.0)
        for table in list(getattr(page, "tables", None) or []):
            if not getattr(table, "pages", None):
                try:
                    table.pages = [int(getattr(table, "page", 0) or page.page_number)]
                except Exception:
                    table.pages = [int(getattr(page, "page_number", 1) or 1)]
            setattr(table, "_page_height", page_h)

    # Walk pages in order; for each table on page N, try to absorb the first
    # compatible continuation table on page N+1 (same ordinal preference).
    for idx in range(len(page_list) - 1):
        cur_page = page_list[idx]
        next_page = page_list[idx + 1]
        cur_tables = list(getattr(cur_page, "tables", None) or [])
        next_tables = list(getattr(next_page, "tables", None) or [])
        if not cur_tables or not next_tables:
            continue
        cur_h = float((getattr(cur_page, "metrics", None) or {}).get("page_height") or 792.0)
        next_h = float((getattr(next_page, "metrics", None) or {}).get("page_height") or 792.0)

        consumed: set[int] = set()
        for t_idx, prev in enumerate(cur_tables):
            prev_headers = list(getattr(prev, "headers", []) or [])
            prev_rows = [list(r) for r in (getattr(prev, "rows", []) or [])]
            best_j = -1
            for j, nxt in enumerate(next_tables):
                if j in consumed:
                    continue
                next_headers = list(getattr(nxt, "headers", []) or [])
                next_rows = [list(r) for r in (getattr(nxt, "rows", []) or [])]
                # (a) column counts
                if not _cols_match(
                    [prev_headers] + prev_rows if prev_headers else prev_rows,
                    [next_headers] + next_rows if next_headers else next_rows,
                ):
                    # Also accept equal header lengths directly.
                    if len(prev_headers) != len(next_headers) and not (
                        next_headers and len(prev_headers) == max(len(r) for r in next_rows or [[]])
                    ):
                        if len(prev_headers) != max((len(r) for r in next_rows), default=0):
                            continue
                # (b) repeated / missing headers
                _, stripped_rows, ok_headers = _strip_repeated_header_row(
                    prev_headers, next_headers, next_rows
                )
                if not ok_headers:
                    continue
                # (c) bbox continuation
                if not _looks_continuation_bbox(
                    prev, nxt, page_height_prev=cur_h, page_height_next=next_h
                ):
                    continue
                best_j = j
                # Merge into prev.
                prev.rows = prev_rows + stripped_rows
                pages_acc = list(getattr(prev, "pages", None) or [getattr(prev, "page", cur_page.page_number)])
                nxt_pages = list(getattr(nxt, "pages", None) or [getattr(nxt, "page", next_page.page_number)])
                for p in nxt_pages:
                    try:
                        pages_acc.append(int(p))
                    except (TypeError, ValueError):
                        pass
                prev.pages = sorted(set(int(p) for p in pages_acc))
                # Carry low-confidence / merged cells with row offset.
                offset = len(prev_rows)
                for item in list(getattr(nxt, "low_confidence_cells", None) or []):
                    if isinstance(item, dict):
                        adj = dict(item)
                        if "row" in adj and adj["row"] is not None:
                            try:
                                adj["row"] = int(adj["row"]) + offset
                            except (TypeError, ValueError):
                                pass
                        prev.low_confidence_cells = list(
                            getattr(prev, "low_confidence_cells", None) or []
                        ) + [adj]
                for item in list(getattr(nxt, "merged_cells", None) or []):
                    if isinstance(item, dict):
                        adj = dict(item)
                        if adj.get("row_index") is not None:
                            try:
                                adj["row_index"] = int(adj["row_index"]) + offset
                            except (TypeError, ValueError):
                                pass
                        prev.merged_cells = list(getattr(prev, "merged_cells", None) or []) + [adj]
                # Prefer richer caption from the first page.
                if not getattr(prev, "caption", "") and getattr(nxt, "caption", ""):
                    prev.caption = nxt.caption
                consumed.add(j)
                break
            if best_j < 0:
                continue
        if consumed:
            next_page.tables = [
                t for j, t in enumerate(next_tables) if j not in consumed
            ]

    flat: List[Any] = []
    for page in page_list:
        for table in list(getattr(page, "tables", None) or []):
            if not getattr(table, "pages", None):
                table.pages = [int(getattr(table, "page", page.page_number) or 1)]
            flat.append(table)
    return flat


def repair_table_data(
    table: Any,
    *,
    filepath: Optional[str] = None,
    caption: str = "",
) -> Any:
    """In-place structural normalize of a TableData-like object.

    ``filepath`` is retained for call-site compatibility but is unused — Camelot
    repair has been removed. Collapsed grids are normalized in-place and flagged
    via linearized facts / existing quality metadata.
    """
    del filepath  # no external repair engine
    headers = list(getattr(table, "headers", []) or [])
    rows = [list(r) for r in (getattr(table, "rows", []) or [])]

    if looks_collapsed_table(headers, rows) and rows:
        headers, rows = _promote_range_headers(
            [headers] + rows if headers else rows
        )

    normalized = normalize_table_grid(
        headers,
        rows,
        caption=caption or getattr(table, "caption", "") or "",
    )
    if normalized.rows and not looks_collapsed_table(normalized.headers, normalized.rows):
        table.headers = normalized.headers
        table.rows = normalized.rows
    elif normalized.rows:
        # Still apply schedule-fragment fills even when the grid stays sparse.
        table.headers = normalized.headers
        table.rows = normalized.rows
    setattr(table, "linearized_facts", normalized.facts)
    return table
