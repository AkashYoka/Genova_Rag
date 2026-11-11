"""Domain-agnostic deterministic table / matrix lookups for RAG.

Resolves row×column cells from retrieved chunks using query parameters
extracted from the question. The LLM explains these verified results; it
must not invent the cell value.

No product names, indications, or domain-specific rules are hardcoded.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

VERIFIED_BLOCK_MARKER = "Table lookup evidence"

_RANGE_RE = re.compile(
    r"(?:≥\s*|>=\s*|>\s*)?(?P<a>\d+(?:\.\d+)?)\s*[−\-–—to]+\s*"
    r"(?:≥\s*|>=\s*|>\s*)?(?P<b>\d+(?:\.\d+)?)",
    re.I,
)
_GTE_RE = re.compile(r"(?:≥|>=|>)\s*(?P<v>\d+(?:\.\d+)?)", re.I)
_LTE_RE = re.compile(r"(?:≤|<=|<)\s*(?P<v>\d+(?:\.\d+)?)", re.I)
_NUM_UNIT_RE = re.compile(
    r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>[A-Za-z%µμ/°]+)?",
    re.I,
)
_STATED_QTY_SCHED_RE = re.compile(
    r"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z%µμ/°]+)?\s+"
    r"(?:(?:subcutaneously|sc|orally|iv|im)\s+)?"
    r"(?:every|q)\s*(?P<n>\d+)\s*(?P<period>weeks?|days?|months?|hours?|hrs?)",
    re.I,
)
_TABLE_FACTS_LINE_RE = re.compile(
    r"^\s*[-*•]?\s*(?P<body>.+?)\s*$",
)
_MD_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")


@dataclass(frozen=True)
class QueryParameter:
    value: float
    unit: str
    raw: str
    start: int
    end: int


@dataclass
class ParsedTable:
    headers: List[str]
    rows: List[List[str]]
    facts: List[str] = field(default_factory=list)
    source_index: int = 0
    filename: str = ""
    page: Any = "?"
    origin: str = "unknown"


@dataclass(frozen=True)
class CellLookup:
    row_label: str
    col_header: str
    value: str
    frequency: str
    filename: str
    page: Any
    source_index: int
    confidence: float
    row_param: str
    col_param: str


@dataclass(frozen=True)
class StatedComparison:
    stated: str
    looked_up: str
    matches: bool
    detail: str


@dataclass
class VerifiedEvidence:
    lookups: List[CellLookup] = field(default_factory=list)
    comparisons: List[StatedComparison] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.lookups or self.comparisons)

    def format_block(self) -> str:
        """Natural evidence brief for the LLM (no 'deterministic' / code jargon)."""
        if not self.has_content:
            return ""
        lines = [
            f"## {VERIFIED_BLOCK_MARKER}",
            "Authoritative matched cell from the document table (already resolved "
            "for the user's numeric parameters). Do not pick a different row or "
            "column from other excerpts.",
        ]
        for lu in self.lookups:
            freq = f", frequency {lu.frequency}" if lu.frequency else ""
            lines.append(
                f"- Recommended table value: {lu.value}{freq} "
                f"(row band {lu.row_label!r}, column band {lu.col_header!r}; "
                f"query axes {lu.row_param!r} × {lu.col_param!r}). "
                f"[Source: {lu.filename}, Page {lu.page}]"
            )
        for cmp_ in self.comparisons:
            if cmp_.matches:
                lines.append(
                    f"- Stated {cmp_.stated!r} matches the table ({cmp_.looked_up!r})."
                )
            else:
                lines.append(
                    f"- MISMATCH: stated {cmp_.stated!r} vs table {cmp_.looked_up!r} "
                    f"({cmp_.detail}). The stated regimen is NOT the table-recommended "
                    f"value; describe this as a dosing/schedule mismatch (special "
                    f"situation / inappropriate dose), not as a match."
                )
        return "\n".join(lines)

    def matched_pages(self) -> set:
        pages: set = set()
        for lu in self.lookups:
            try:
                pages.add(int(lu.page))
            except (TypeError, ValueError):
                pages.add(lu.page)
        return pages


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _norm_unit(unit: Optional[str]) -> str:
    u = (unit or "").strip().lower()
    aliases = {
        "iu/ml": "iu/ml",
        "iu": "iu",
        "kg": "kg",
        "mg": "mg",
        "g": "g",
        "ml": "ml",
        "lbs": "lb",
        "lb": "lb",
        "%": "%",
    }
    return aliases.get(u, u)


_AGE_UNIT = {"year", "years", "yr", "yrs"}
_SCHEDULE_UNIT = {"week", "weeks", "day", "days", "month", "months", "hour", "hours", "hr", "hrs"}


def extract_query_parameters(question: str) -> List[QueryParameter]:
    """Extract numeric parameters (with optional units) from the question.

    Stated quantity+schedule phrases (e.g. \"15 every 4 weeks\") are excluded
    from axis matching — they are comparison targets, not row/column keys.
    """
    q = question or ""
    stated_spans = [(m.start(), m.end()) for m in _STATED_QTY_SCHED_RE.finditer(q)]
    params: List[QueryParameter] = []
    seen: set[Tuple[float, str, int]] = set()
    for m in _NUM_UNIT_RE.finditer(q):
        try:
            value = float(m.group("v"))
        except (TypeError, ValueError):
            continue
        unit = _norm_unit(m.group("u"))
        key = (value, unit, m.start())
        if key in seen:
            continue
        # Skip numbers inside stated qty+schedule (prescribed/quoted amounts).
        if any(s <= m.start() < e for s, e in stated_spans):
            continue
        window = q[max(0, m.start() - 16) : m.end() + 20].lower()
        # Skip schedule cadence ("every 4 weeks") — not a lookup axis.
        if re.search(r"\bevery\s+" + re.escape(m.group("v")) + r"\b", window):
            continue
        if re.search(r"\bq\s*" + re.escape(m.group("v")) + r"\b", window):
            continue
        if unit in _SCHEDULE_UNIT:
            continue
        # Skip ages: "35-year-old", "35 years old".
        if re.search(
            r"\b" + re.escape(m.group("v")) + r"\s*[- ]?\s*(year|yr)s?\s*-?\s*old\b",
            window,
        ):
            continue
        if unit in _AGE_UNIT:
            continue
        seen.add(key)
        params.append(
            QueryParameter(
                value=value,
                unit=unit,
                raw=_norm(m.group(0)),
                start=m.start(),
                end=m.end(),
            )
        )
    params.sort(key=lambda p: (0 if p.unit else 1, p.start))
    return params


def _split_range_labels(label: str) -> List[str]:
    """Split a possibly smashed header into discrete range labels."""
    text = _norm(label)
    if not text:
        return []
    matches = [_norm(m.group(0)) for m in _RANGE_RE.finditer(text)]
    if len(matches) >= 2:
        # Preserve trailing unit on each piece when the smashed cell ends with a unit.
        unit_m = re.search(r"([A-Za-z%µμ/°]+)\s*$", text)
        unit = unit_m.group(1) if unit_m else ""
        out = []
        for piece in matches:
            if unit and not re.search(r"[A-Za-z%µμ/°]", piece):
                out.append(f"{piece} {unit}")
            else:
                out.append(piece)
        return out
    return [text] if text else []


def _parse_range_bounds(label: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    text = _norm(label)
    if not text:
        return None
    m = _RANGE_RE.search(text)
    if m:
        a, b = float(m.group("a")), float(m.group("b"))
        lo, hi = min(a, b), max(a, b)
        # Open-ended prefixes like ">300-400" mean (300, 400] in many matrices.
        if re.match(r"^[>≥>]", text):
            return (a if text.lstrip().startswith((">", "≥")) else lo, hi)
        return (lo, hi)
    m = _GTE_RE.search(text)
    if m:
        return (float(m.group("v")), None)
    m = _LTE_RE.search(text)
    if m:
        return (None, float(m.group("v")))
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(nums) == 1:
        v = float(nums[0])
        return (v, v)
    return None


def _bounds_contain(bounds: Tuple[Optional[float], Optional[float]], value: float) -> bool:
    lo, hi = bounds
    if lo is not None and value < lo - 1e-9:
        # Allow inclusive lower for open ">X-Y" style by treating exact lo as in-range
        # when the label used a range form (handled by caller). Use strict for ≥ only.
        return False
    if hi is not None and value > hi + 1e-9:
        return False
    return True


def _score_bounds_match(
    bounds: Tuple[Optional[float], Optional[float]],
    param: QueryParameter,
    label_unit: str,
) -> float:
    lo, hi = bounds
    score = 1.0
    if lo is not None and hi is not None and lo != hi:
        width = hi - lo
        score += max(0.0, 0.35 - min(width, 400) / 1200.0)
    elif lo == hi:
        score += 0.25
    if param.unit and label_unit and param.unit == label_unit:
        score += 0.45
    elif param.unit and label_unit and param.unit != label_unit:
        score -= 0.25
    elif param.unit and not label_unit:
        score += 0.05
    elif not param.unit:
        score -= 0.35  # bare numbers (e.g. age) are weak axis candidates
    return score


def _label_contains_value(label: str, param: QueryParameter) -> Tuple[bool, float]:
    """Return (matches, score) if param falls in any range segment of the label."""
    text = _norm(label)
    if not text:
        return False, 0.0
    label_unit = ""
    um = re.search(r"([A-Za-z%µμ/°]+)\s*$", text)
    if um:
        label_unit = _norm_unit(um.group(1))
    # Also detect unit tokens mid-label (kg, iu/ml).
    for tok in re.findall(r"[A-Za-z%µμ/°]+", text):
        nu = _norm_unit(tok)
        if nu in {"kg", "mg", "g", "ml", "iu", "iu/ml", "%", "lb"}:
            label_unit = label_unit or nu

    best = 0.0
    matched = False
    for piece in _split_range_labels(text):
        bounds = _parse_range_bounds(piece)
        if bounds is None:
            continue
        lo, hi = bounds
        # Inclusive range containment for matrix bands.
        in_range = True
        if lo is not None and param.value < lo - 1e-9:
            in_range = False
        if hi is not None and param.value > hi + 1e-9:
            in_range = False
        # Common table convention: ">300−400" includes 350 and often 300+.
        if not in_range and lo is not None and hi is not None:
            if abs(param.value - lo) < 1e-9 or abs(param.value - hi) < 1e-9:
                in_range = True
        if not in_range:
            continue
        matched = True
        best = max(best, _score_bounds_match(bounds, param, label_unit))

    if matched:
        return True, best
    if param.unit and param.unit in text.lower():
        token = str(int(param.value)) if float(param.value).is_integer() else str(param.value)
        if token in text:
            return True, 0.4
    return False, 0.0


def _split_cell_values(value: str) -> List[str]:
    text = _norm(value)
    if not text:
        return []
    # Multi-value smashed cells: "150 300" or "150 / 300".
    parts = re.split(r"[/,;]|\\s{2,}", text)
    parts = [_norm(p) for p in parts if _norm(p)]
    if len(parts) >= 2:
        return parts
    # Space-separated pure numbers.
    nums = re.findall(r"\d+(?:\.\d+)?(?:\s*[A-Za-z%]+)?", text)
    if len(nums) >= 2 and len("".join(nums)) >= len(text.replace(" ", "")) - 2:
        return [_norm(n) for n in nums]
    return [text]


def _parse_table_json(raw: Any) -> Optional[Tuple[List[str], List[List[str]]]]:
    if raw is None or raw == "":
        return None
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(data, dict):
        return None
    headers = data.get("headers") or data.get("columns") or []
    rows = data.get("rows") or data.get("body") or []
    if not isinstance(headers, list) or not isinstance(rows, list):
        return None
    hdrs = [_norm(h) for h in headers]
    body = [[_norm(c) for c in (row if isinstance(row, (list, tuple)) else [row])] for row in rows]
    if not body:
        return None
    return hdrs, body


def _parse_table_facts_block(text: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """Build a sparse grid from linearized 'Table facts:' lines when possible."""
    if "table facts:" not in (text or "").lower():
        return [], [], []
    lower = text.lower()
    idx = lower.find("table facts:")
    block = text[idx + len("table facts:") :]
    # Stop at next major markdown heading if any.
    stop = re.search(r"\n#{1,3}\s|\n\[?\d+\]\s+Source:", block)
    if stop:
        block = block[: stop.start()]
    facts: List[str] = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _TABLE_FACTS_LINE_RE.match(line)
        body = (m.group("body") if m else line).strip()
        if body.lower().startswith("table facts"):
            continue
        if body.startswith("-"):
            body = body[1:].strip()
        if "|" not in body:
            continue
        facts.append(body)

    # Reconstruct headers/rows from facts: row | col | value [| freq]
    col_headers: List[str] = []
    row_map: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for fact in facts:
        # Drop optional caption prefix "caption: "
        if ":" in fact.split("|", 1)[0] and fact.lower().count("|") >= 2:
            # Keep as-is; caption may be first segment without pipes only.
            pass
        parts = [_norm(p) for p in fact.split("|")]
        if len(parts) < 3:
            continue
        # Handle "caption: row | col | value"
        if ":" in parts[0] and len(parts) >= 3:
            maybe = parts[0].split(":", 1)
            if len(maybe) == 2 and _norm(maybe[1]):
                parts = [_norm(maybe[1])] + parts[1:]
        row_label, col_header, value = parts[0], parts[1], parts[2]
        freq = parts[3] if len(parts) > 3 else ""
        if col_header not in col_headers:
            col_headers.append(col_header)
        row_map.setdefault(row_label, {})[col_header] = (value, freq)

    if not row_map or not col_headers:
        return [], [], facts

    headers = ["Row label"] + col_headers
    # Detect a shared frequency column pattern.
    rows: List[List[str]] = []
    for row_label, cols in row_map.items():
        freqs = {f for (_v, f) in cols.values() if f}
        freq = next(iter(freqs)) if len(freqs) == 1 else ""
        row = [row_label]
        if freq:
            # Insert frequency as column 2 if not already a header.
            pass
        for ch in col_headers:
            cell = cols.get(ch)
            row.append(cell[0] if cell else "")
        if freq:
            row.append(freq)
        rows.append(row)
    if any(len(r) > len(headers) for r in rows):
        headers = headers + ["Frequency"]
        for r in rows:
            while len(r) < len(headers):
                r.append("")
    return headers, rows, facts


def _parse_markdown_table(text: str) -> Optional[Tuple[List[str], List[List[str]]]]:
    rows_raw: List[List[str]] = []
    for line in (text or "").splitlines():
        m = _MD_ROW_RE.match(line)
        if not m:
            continue
        cells = [_norm(c) for c in m.group(1).split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells):
            continue
        rows_raw.append(cells)
    if len(rows_raw) < 2:
        return None
    headers = rows_raw[0]
    body = rows_raw[1:]
    width = max(len(headers), max((len(r) for r in body), default=0))
    headers = (headers + [""] * width)[:width]
    body = [(r + [""] * width)[:width] for r in body]
    return headers, body


def tables_from_chunk(chunk: Dict[str, Any], source_index: int) -> List[ParsedTable]:
    """Extract one or more parsed tables from a retrieved chunk."""
    text = chunk.get("text") or ""
    filename = chunk.get("filename") or "document"
    page = chunk.get("page", "?")
    out: List[ParsedTable] = []

    parsed_json = _parse_table_json(chunk.get("table_json"))
    if parsed_json:
        headers, rows = parsed_json
        out.append(
            ParsedTable(
                headers=headers,
                rows=rows,
                source_index=source_index,
                filename=filename,
                page=page,
                origin="table_json",
            )
        )

    hdrs, rows, facts = _parse_table_facts_block(text)
    if rows:
        out.append(
            ParsedTable(
                headers=hdrs,
                rows=rows,
                facts=facts,
                source_index=source_index,
                filename=filename,
                page=page,
                origin="table_facts",
            )
        )
    elif facts:
        # Facts without reconstructable grid — still usable via fact lines.
        out.append(
            ParsedTable(
                headers=[],
                rows=[],
                facts=facts,
                source_index=source_index,
                filename=filename,
                page=page,
                origin="table_facts_lines",
            )
        )

    if not out:
        md = _parse_markdown_table(text)
        if md:
            headers, rows = md
            out.append(
                ParsedTable(
                    headers=headers,
                    rows=rows,
                    source_index=source_index,
                    filename=filename,
                    page=page,
                    origin="markdown",
                )
            )
    return out


def _guess_freq_col(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> int:
    for i, h in enumerate(headers or []):
        if re.search(r"freq|interval|schedule|every", h or "", re.I):
            return i
    if not rows:
        return -1
    width = max((len(r) for r in rows), default=0)
    best_i, best = -1, 0
    for c in range(min(width, 4)):
        score = 0
        for row in rows:
            if c < len(row) and re.search(
                r"(?i)every\s+\d+|weeks?|days?|months?", row[c] or ""
            ):
                score += 1
        if score > best:
            best, best_i = score, c
    return best_i if best else -1


def _lookup_from_grid(
    table: ParsedTable,
    params: Sequence[QueryParameter],
) -> Optional[CellLookup]:
    if not table.rows or len(params) < 1:
        return None
    width = max([len(table.headers)] + [len(r) for r in table.rows] or [0])
    if width < 2:
        return None
    headers = (list(table.headers) + [f"Column {i+1}" for i in range(width)])[:width]
    headers = [h if h else f"Column {i+1}" for i, h in enumerate(headers)]
    freq_col = _guess_freq_col(headers, table.rows)

    # Score each param against each row label (col 0) and each data column header.
    row_scores: List[Tuple[float, int, QueryParameter, str]] = []
    for r_i, row in enumerate(table.rows):
        cells = list(row) + [""] * (width - len(row))
        label = _norm(cells[0])
        for p in params:
            ok, score = _label_contains_value(label, p)
            if ok:
                row_scores.append((score, r_i, p, label))

    col_scores: List[Tuple[float, int, QueryParameter, str]] = []
    for c in range(1, width):
        if c == freq_col:
            continue
        header = headers[c]
        for p in params:
            ok, score = _label_contains_value(header, p)
            if ok:
                col_scores.append((score, c, p, header))

    if not row_scores:
        return None

    row_scores.sort(key=lambda x: x[0], reverse=True)

    # Prefer two distinct parameters for row and column when available.
    if col_scores:
        col_scores.sort(key=lambda x: x[0], reverse=True)
        best: Optional[Tuple[float, int, int, QueryParameter, QueryParameter, str, str]] = None
        for r_score, r_i, r_p, r_label in row_scores[:8]:
            for c_score, c_i, c_p, c_label in col_scores[:8]:
                if r_p.start == c_p.start and r_p.value == c_p.value and r_p.unit == c_p.unit:
                    continue
                combined = r_score + c_score + (0.2 if r_p.unit != c_p.unit else 0.0)
                if best is None or combined > best[0]:
                    best = (combined, r_i, c_i, r_p, c_p, r_label, c_label)
        if best is None:
            return None
        _combined, r_i, c_i, r_p, c_p, r_label, c_label = best
        cells = list(table.rows[r_i]) + [""] * (width - len(table.rows[r_i]))
        raw_value = _norm(cells[c_i])
        if not raw_value:
            return None
        value = _disambiguate_value(c_label, c_p, raw_value)
        freq = _norm(cells[freq_col]) if 0 <= freq_col < width else ""
        if freq_col == 1 and not re.search(r"(?i)every\s+\d+", freq):
            # Some grids put frequency in column 1; keep only schedule-like text.
            for cell in cells[1:]:
                if re.search(r"(?i)every\s+\d+", cell or ""):
                    freq = _norm(cell)
                    break
        lu = CellLookup(
            row_label=r_label,
            col_header=c_label,
            value=value,
            frequency=freq,
            filename=table.filename,
            page=table.page,
            source_index=table.source_index,
            confidence=min(1.0, (_combined + 0.5) / 3.0),
            row_param=r_p.raw,
            col_param=c_p.raw,
        )
        return lu

    # Single-axis: best row, first non-empty data cell.
    r_score, r_i, r_p, r_label = row_scores[0]
    cells = list(table.rows[r_i]) + [""] * (width - len(table.rows[r_i]))
    for c in range(1, width):
        if c == freq_col:
            continue
        value = _norm(cells[c])
        if value:
            freq = _norm(cells[freq_col]) if 0 <= freq_col < width else ""
            return CellLookup(
                row_label=r_label,
                col_header=headers[c],
                value=value,
                frequency=freq,
                filename=table.filename,
                page=table.page,
                source_index=table.source_index,
                confidence=min(1.0, r_score / 1.5),
                row_param=r_p.raw,
                col_param="(single-axis)",
            )
    return None


def _disambiguate_value(col_header: str, col_param: QueryParameter, value: str) -> str:
    """If a smashed header has N ranges and the cell has N values, pick the matching one."""
    ranges = _split_range_labels(col_header)
    values = _split_cell_values(value)
    if len(ranges) >= 2 and len(values) == len(ranges):
        for i, piece in enumerate(ranges):
            ok, _ = _label_contains_value(piece, col_param)
            if ok:
                return values[i]
    if len(values) == 1:
        return values[0]
    # Prefer a single numeric token when present.
    numeric = [v for v in values if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*[A-Za-z%]+)?", v)]
    if len(numeric) == 1:
        return numeric[0]
    return value


def _lookup_quality_bonus(lu: CellLookup) -> float:
    bonus = 0.0
    if lu.row_param and re.search(r"[A-Za-z]", lu.row_param):
        bonus += 0.25
    if lu.col_param and re.search(r"[A-Za-z]", lu.col_param):
        bonus += 0.25
    # Prefer assay/weight-style axis units over amount units on the row key.
    if lu.row_param and re.search(r"(?i)\b(iu(?:/ml)?|kg|%|lb)\b", lu.row_param):
        bonus += 0.35
    if lu.col_param and re.search(r"(?i)\b(kg|lb|iu(?:/ml)?|%)\b", lu.col_param):
        bonus += 0.35
    if lu.row_param and re.search(r"(?i)\bmg\b", lu.row_param):
        bonus -= 0.45
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*[A-Za-z%]+)?", _norm(lu.value)):
        bonus += 0.35
    elif len(_split_cell_values(lu.value)) >= 2:
        bonus -= 0.4
    if lu.frequency and re.search(r"(?i)every\s+\d+", lu.frequency):
        bonus += 0.15
    if re.search(r"(?i)column\s+\d+|freq\.?", lu.col_header):
        bonus -= 0.5
    if re.search(r"(?i)^freq\.?$", _norm(lu.frequency)):
        bonus -= 0.35
    if re.search(r"(?i)insufficient|recommend a dose", lu.value):
        bonus -= 0.8
    # Prefer two-axis matches over single-axis fallbacks.
    if lu.col_param == "(single-axis)":
        bonus -= 0.6
    return bonus


def _lookup_from_fact_lines(
    table: ParsedTable,
    params: Sequence[QueryParameter],
) -> Optional[CellLookup]:
    if not table.facts or len(params) < 2:
        return None
    best: Optional[Tuple[float, CellLookup]] = None
    for fact in table.facts:
        parts = [_norm(p) for p in fact.split("|")]
        if len(parts) < 3:
            continue
        if ":" in parts[0]:
            _left, right = parts[0].split(":", 1)
            if _norm(right):
                parts = [_norm(right)] + parts[1:]
        row_label, col_header, value = parts[0], parts[1], parts[2]
        freq = parts[3] if len(parts) > 3 else ""
        if re.search(r"(?i)^column\s+\d+$", col_header):
            continue
        row_hits = []
        col_hits = []
        for p in params:
            ok, score = _label_contains_value(row_label, p)
            if ok:
                row_hits.append((score, p))
            ok, score = _label_contains_value(col_header, p)
            if ok:
                col_hits.append((score, p))
        if not row_hits or not col_hits:
            continue
        row_hits.sort(key=lambda x: x[0], reverse=True)
        col_hits.sort(key=lambda x: x[0], reverse=True)
        r_score, r_p = row_hits[0]
        c_score, c_p = next(
            (
                (s, p)
                for s, p in col_hits
                if not (p.start == r_p.start and abs(p.value - r_p.value) < 1e-9)
            ),
            (0.0, None),  # type: ignore
        )
        if c_p is None:
            continue
        resolved_value = _disambiguate_value(col_header, c_p, value)
        if not resolved_value or not re.search(r"\d", resolved_value):
            continue
        combined = r_score + c_score
        lu = CellLookup(
            row_label=row_label,
            col_header=col_header,
            value=resolved_value,
            frequency=freq if re.search(r"(?i)every\s+\d+", freq) else freq,
            filename=table.filename,
            page=table.page,
            source_index=table.source_index,
            confidence=min(1.0, (combined + 0.5) / 3.0),
            row_param=r_p.raw,
            col_param=c_p.raw,
        )
        combined += _lookup_quality_bonus(lu)
        if best is None or combined > best[0]:
            best = (combined, lu)
    return best[1] if best else None


def _normalize_schedule(text: str) -> str:
    t = _norm(text).lower()
    t = re.sub(r"\bq\s*(\d+)\s*(w|wk|weeks?)\b", r"every \1 weeks", t)
    t = re.sub(r"\bq\s*(\d+)\s*(d|days?)\b", r"every \1 days", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _extract_stated_qty_schedules(question: str) -> List[Tuple[str, str, str]]:
    """Return list of (qty_raw, schedule_raw, combined)."""
    out: List[Tuple[str, str, str]] = []
    for m in _STATED_QTY_SCHED_RE.finditer(question or ""):
        qty = _norm(f"{m.group('qty')} {m.group('unit') or ''}")
        sched = _norm(f"every {m.group('n')} {m.group('period')}")
        out.append((qty, sched, f"{qty} {sched}"))
    return out


def _values_equivalent(a: str, b: str) -> bool:
    na, nb = _normalize_schedule(a), _normalize_schedule(b)
    if na == nb:
        return True
    # Compare leading numbers + units loosely.
    ma = _NUM_UNIT_RE.search(na)
    mb = _NUM_UNIT_RE.search(nb)
    if ma and mb:
        try:
            if abs(float(ma.group("v")) - float(mb.group("v"))) < 1e-9:
                ua, ub = _norm_unit(ma.group("u")), _norm_unit(mb.group("u"))
                if not ua or not ub or ua == ub:
                    # Also compare schedule fragments if both have them.
                    sa = re.search(r"every\s+\d+\s+\w+", na)
                    sb = re.search(r"every\s+\d+\s+\w+", nb)
                    if sa and sb:
                        return sa.group(0) == sb.group(0)
                    if not sa and not sb:
                        return True
        except (TypeError, ValueError):
            pass
    return False


def compare_stated_to_lookups(
    question: str,
    lookups: Sequence[CellLookup],
) -> List[StatedComparison]:
    if not lookups:
        return []
    stated = _extract_stated_qty_schedules(question)
    if not stated:
        return []
    comparisons: List[StatedComparison] = []
    for qty, sched, combined in stated:
        for lu in lookups:
            looked = _norm(f"{lu.value} {lu.frequency}".strip())
            stated_full = combined
            # Compare quantity to cell value and schedule to frequency when present.
            qty_match = _values_equivalent(qty, lu.value)
            sched_match = True
            if lu.frequency:
                sched_match = _values_equivalent(sched, lu.frequency)
            matches = qty_match and sched_match
            if matches:
                detail = "Stated quantity/schedule matches the looked-up cell."
            else:
                bits = []
                if not qty_match:
                    bits.append(f"quantity {qty!r} ≠ table value {lu.value!r}")
                if not sched_match:
                    bits.append(f"schedule {sched!r} ≠ table frequency {lu.frequency!r}")
                detail = "; ".join(bits) if bits else "Stated values differ from table cell."
            comparisons.append(
                StatedComparison(
                    stated=stated_full,
                    looked_up=looked,
                    matches=matches,
                    detail=detail,
                )
            )
            break  # one comparison per stated phrase against best lookup
    return comparisons


def build_verified_evidence(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    min_confidence: float = 0.45,
    max_lookups: int = 3,
) -> VerifiedEvidence:
    """Deterministically resolve table cells for the question from retrieved chunks."""
    params = extract_query_parameters(question)
    if not params or not chunks:
        return VerifiedEvidence()

    candidates: List[CellLookup] = []
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        for table in tables_from_chunk(chunk, source_index=i + 1):
            lu = None
            if table.rows:
                lu = _lookup_from_grid(table, params)
            if lu is None and table.facts:
                lu = _lookup_from_fact_lines(table, params)
            if lu is not None and lu.confidence >= min_confidence:
                candidates.append(lu)

    # Rank by quality (unit-bearing axes + clean cell) then confidence.
    ranked = sorted(
        candidates,
        key=lambda x: (x.confidence + _lookup_quality_bonus(x), x.confidence),
        reverse=True,
    )
    has_two_axis = any(lu.col_param != "(single-axis)" for lu in ranked)
    uniq: List[CellLookup] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for lu in ranked:
        if has_two_axis and lu.col_param == "(single-axis)":
            continue
        if lu.confidence + _lookup_quality_bonus(lu) < 0.6:
            continue
        key = (lu.row_label, lu.col_header, lu.value, lu.frequency)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(lu)
        if len(uniq) >= max_lookups:
            break

    comparisons = compare_stated_to_lookups(question, uniq)
    evidence = VerifiedEvidence(lookups=uniq, comparisons=comparisons)
    if evidence.has_content:
        logger.info(
            "Structured lookup resolved %d cell(s), %d comparison(s)",
            len(evidence.lookups),
            len(evidence.comparisons),
        )
    return evidence


def attach_verified_evidence(context: str, evidence: VerifiedEvidence) -> str:
    """Place table-lookup evidence first so the model sees it before noisy tables."""
    block = evidence.format_block()
    if not block:
        return context or ""
    if context and context.strip():
        return f"{block}\n\n{context}"
    return block


def focus_chunks_for_evidence(
    chunks: Sequence[Dict[str, Any]],
    evidence: VerifiedEvidence,
    *,
    max_extra: int = 3,
) -> List[Dict[str, Any]]:
    """Prefer chunks from matched lookup pages; drop conflicting matrix noise."""
    if not evidence.lookups or not chunks:
        return list(chunks)
    pages = evidence.matched_pages()
    source_indexes = {lu.source_index for lu in evidence.lookups}
    primary: List[Dict[str, Any]] = []
    secondary: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        page = chunk.get("page")
        try:
            page_i = int(page)
        except (TypeError, ValueError):
            page_i = page
        if (i + 1) in source_indexes or page_i in pages:
            primary.append(chunk)
        else:
            # Keep a few non-table narrative chunks for AE / counseling context.
            text = (chunk.get("text") or "").lower()
            is_matrix = "table facts:" in text or bool(chunk.get("table_json"))
            if not is_matrix:
                secondary.append(chunk)
    focused = primary + secondary[:max_extra]
    return focused or list(chunks)[: max(1, max_extra)]


# Back-compat alias
prepend_verified_evidence = attach_verified_evidence
