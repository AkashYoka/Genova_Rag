"""Generic matrix/table repair and retrieval-friendly linearization.

Complex PDF tables often use multi-row headers, merged cells, and labels split
across rows (e.g. Every / 4 / weeks). Native word-alignment extraction can smash
row labels into one cell and break row×column relationships. This module:

1. Detects collapsed / low-quality grids.
2. Optionally repairs a page via Camelot (stream/lattice/network).
3. Forward-fills fragmented frequency/schedule labels.
4. Emits linearized cell facts for retrieval and LLM context.

Domain-agnostic: works for any row-label × column-header numeric matrix
(schedules, rate cards, invoices, lab grids, etc.).
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
_NUMERIC_DOSE_RE = re.compile(r"^\d+(?:\.\d+)?(?:\s*mg)?$", re.I)


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
        if any(_NUMERIC_DOSE_RE.match(normalize_ws(c)) for c in row[1:]):
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

    Camelot sometimes merges adjacent weight columns into one cell with newlines.
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


def _promote_weight_headers(raw_rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """Prefer a row that looks like body-weight column headers over a sparse title row."""
    if len(raw_rows) < 2:
        return raw_rows[0], raw_rows[1:] if len(raw_rows) > 1 else []

    def weight_score(row: Sequence[str]) -> int:
        score = 0
        for cell in row:
            text = normalize_ws(cell)
            if re.search(r"(?i)\bkg\b", text):
                score += 2
            score += len(split_range_labels(text))
        return score

    best_i = 0
    best_score = weight_score(raw_rows[0])
    # Only look at the first few rows for header promotion.
    for i, row in enumerate(raw_rows[:4]):
        score = weight_score(row)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i == 0 or best_score < 2:
        return raw_rows[0], raw_rows[1:]
    headers = list(raw_rows[best_i])
    # Keep useful labels from earlier header rows when a cell is empty.
    for prior in raw_rows[:best_i]:
        for c, cell in enumerate(prior):
            if c >= len(headers):
                break
            if not normalize_ws(headers[c]) and normalize_ws(cell):
                headers[c] = normalize_ws(cell)
    if not normalize_ws(headers[0]):
        headers[0] = "Row label"
    body = raw_rows[best_i + 1 :]
    # Drop decorative "Dose (mg)" spacer rows.
    body = [
        row
        for row in body
        if not (
            sum(1 for c in row if normalize_ws(c)) <= 2
            and any(re.search(r"(?i)^dose", c or "") for c in row)
        )
    ]
    return headers, body


def camelot_tables_for_page(
    filepath: str,
    page: int,
    *,
    flavors: Sequence[str] = ("stream", "lattice", "network"),
) -> List[Dict[str, Any]]:
    """Extract tables for a single PDF page via Camelot (best-effort)."""
    try:
        import camelot
    except Exception as exc:
        logger.debug("Camelot unavailable for table repair: %s", exc)
        return []

    records: List[Dict[str, Any]] = []
    page_str = str(int(page))
    for flavor in flavors:
        try:
            try:
                tables = camelot.read_pdf(
                    filepath,
                    pages=page_str,
                    flavor=flavor,
                    suppress_stdout=True,
                )
            except TypeError:
                tables = camelot.read_pdf(filepath, pages=page_str, flavor=flavor)
        except Exception as exc:
            logger.debug("Camelot flavor=%s page=%s failed: %s", flavor, page, exc)
            continue
        for table in tables or []:
            try:
                df = table.df
                raw_rows = [[normalize_ws(str(c)) for c in row] for row in df.values.tolist()]
                raw_rows = [r for r in raw_rows if any(r)]
                if len(raw_rows) < 2:
                    continue
                headers, body = _promote_weight_headers(raw_rows)
                accuracy = None
                report = getattr(table, "parsing_report", None) or {}
                try:
                    if report.get("accuracy") is not None:
                        accuracy = float(report["accuracy"])
                except Exception:
                    pass
                records.append(
                    {
                        "headers": headers,
                        "rows": body,
                        "flavor": flavor,
                        "accuracy": accuracy,
                        "page": page,
                        "collapsed": looks_collapsed_table(headers, body),
                    }
                )
            except Exception as exc:
                logger.debug("Skipping Camelot table parse: %s", exc)
    return records


def pick_best_camelot_repair(
    filepath: str,
    page: int,
    *,
    prefer_unCollapsed: bool = True,
) -> Optional[NormalizedTable]:
    """Choose the best Camelot extraction for a page and normalize it."""
    candidates = camelot_tables_for_page(filepath, page)
    if not candidates:
        return None
    scored: List[Tuple[float, Dict[str, Any], NormalizedTable]] = []
    for cand in candidates:
        norm = normalize_table_grid(
            cand["headers"],
            cand["rows"],
            method=f"camelot:{cand.get('flavor')}",
        )
        score = 0.0
        if prefer_unCollapsed and not looks_collapsed_table(norm.headers, norm.rows):
            score += 50.0
        if cand.get("collapsed"):
            score -= 30.0
        score += min(40.0, float(cand.get("accuracy") or 0.0) / 2.5)
        score += min(30.0, float(len(norm.facts)))
        # Prefer tables with distinct first-column labels.
        labels = {normalize_ws(r[0]) for r in norm.rows if r and normalize_ws(r[0])}
        score += min(20.0, float(len(labels)) * 2.0)
        scored.append((score, cand, norm))
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best_cand, best_norm = scored[0]
    if best_score < 20:
        return None
    best_norm.repaired = True
    best_norm.method = f"camelot:{best_cand.get('flavor')}"
    best_norm.quality_notes = best_norm.quality_notes + ("camelot_repair",)
    return best_norm


def repair_table_data(
    table: Any,
    *,
    filepath: Optional[str] = None,
    caption: str = "",
) -> Any:
    """In-place repair of a TableData-like object when collapsed.

    Returns the same object with headers/rows potentially replaced and
    ``metadata``-style attributes left untouched. Adds ``linearized_facts``
    onto the object when possible.
    """
    headers = list(getattr(table, "headers", []) or [])
    rows = [list(r) for r in (getattr(table, "rows", []) or [])]
    page = int(getattr(table, "page", 0) or 0)

    if looks_collapsed_table(headers, rows) and filepath and page > 0:
        repaired = pick_best_camelot_repair(filepath, page)
        if repaired and not looks_collapsed_table(repaired.headers, repaired.rows):
            table.headers = repaired.headers
            table.rows = repaired.rows
            table.extraction_method = repaired.method
            headers, rows = repaired.headers, repaired.rows

    normalized = normalize_table_grid(headers, rows, caption=caption or getattr(table, "caption", "") or "")
    # Keep normalized frequency fills even when not fully repaired via Camelot.
    if normalized.rows and not looks_collapsed_table(normalized.headers, normalized.rows):
        table.headers = normalized.headers
        table.rows = normalized.rows
    setattr(table, "linearized_facts", normalized.facts)
    return table
