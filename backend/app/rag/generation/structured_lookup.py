"""Domain-agnostic row/column cell lookup against structured table_json.

Matches numeric / range constraints from a question to table labels and
returns a verified intersecting cell, or insufficient evidence when the
structure is ambiguous. No product-, drug-, or indication-specific rules.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_NUM_RE = re.compile(r"(?i)(-?\d+(?:\.\d+)?)\s*([a-zA-Z%/μµ²³]*)")
_RANGE_RE = re.compile(
    r"(?i)"
    r"(≥|>=|>|≤|<=|<)?"
    r"\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*(?:–|-|—|to)\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*([a-zA-Z%/μµ²³]*)?"
)
_SINGLE_BOUND_RE = re.compile(
    r"(?i)(≥|>=|>|≤|<=|<)\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z%/μµ²³]*)?"
)
_FREQ_AFTER_RE = re.compile(
    r"(?i)^\s*(?:weeks?|days?|months?|hours?|hrs?)\b"
)
_AGE_AFTER_RE = re.compile(
    r"(?i)^(?:\s*|-)?(?:years?(?:\s+old)?|-?year(?:s)?(?:-|\s+)?old|-?yo)\b"
)
_COMPARE_HINT_RE = re.compile(
    r"(?i)\b(?:prescrib\w*|receiv\w*|given|administer\w*|order(?:ed)?|"
    r"dispens\w*|intent(?:ionally)?\s+prescrib\w*)\b"
)
_MASS_UNITS = frozenset({"mg", "g", "ug", "µg", "μg", "mcg"})
_WEIGHT_UNITS = frozenset({"kg"})
_ASSAY_UNITS = frozenset(
    {
        "iu/ml",
        "iu/l",
        "ng/ml",
        "ng/l",
        "ug/ml",
        "µg/ml",
        "μg/ml",
        "mcg/ml",
        "mg/l",
        "g/l",
        "mmol/l",
        "umol/l",
        "%",
    }
)
_UNIT_NORMALIZE = {
    "iu/ml": "iu/ml",
    "iu/mL": "iu/ml",
    "iu/l": "iu/l",
    "kg": "kg",
    "mg": "mg",
    "g": "g",
    "ml": "ml",
    "l": "l",
    "%": "%",
    "ug": "ug",
    "µg": "ug",
    "μg": "ug",
    "mcg": "ug",
    "ng/ml": "ng/ml",
    "ug/ml": "ug/ml",
    "µg/ml": "ug/ml",
    "μg/ml": "ug/ml",
    "mcg/ml": "ug/ml",
}

# Constraint roles (domain-agnostic):
# - axis: usable for row/column band matching
# - compare: stated administered/prescribed amount to compare against label cell
# - frequency: schedule token (every N weeks) for row disambiguation only
# - exclude: age / calendar / non-axis identifiers
ROLE_AXIS = "axis"
ROLE_COMPARE = "compare"
ROLE_FREQUENCY = "frequency"
ROLE_EXCLUDE = "exclude"

VERIFIED_BLOCK_MARKER = "CRITICAL — Verified table evidence is authoritative."


@dataclass(frozen=True)
class NumericConstraint:
    value: float
    unit: str = ""
    label_hint: str = ""
    role: str = ROLE_AXIS


@dataclass(frozen=True)
class ParsedRange:
    low: Optional[float]
    high: Optional[float]
    unit: str = ""
    inclusive_low: bool = True
    inclusive_high: bool = True
    raw: str = ""


@dataclass
class VerifiedCell:
    value: str
    row_headers: List[str]
    col_headers: List[str]
    row_index: int
    col_index: int
    page: Optional[int] = None
    filename: str = ""
    confidence: float = 100.0
    table_headers: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        row_s = " | ".join(self.row_headers) if self.row_headers else f"row {self.row_index}"
        col_s = " | ".join(self.col_headers) if self.col_headers else f"col {self.col_index}"
        loc = []
        if self.filename:
            loc.append(self.filename)
        if self.page is not None:
            loc.append(f"Page {self.page}")
        where = ", ".join(loc) if loc else "retrieved table"
        prescribed = self.evidence.get("prescribed_value")
        if prescribed is not None:
            p_unit = self.evidence.get("prescribed_unit") or ""
            p_s = f"{prescribed}{(' ' + p_unit) if p_unit else ''}".strip()
            return (
                f"{VERIFIED_BLOCK_MARKER}\n"
                "This question compares a stated administered/prescribed amount "
                "against a verified table cell.\n"
                f"- Label / table cell value (authoritative): {self.value}\n"
                f"- Stated administered/prescribed amount: {p_s}\n"
                f"- Row: {row_s}\n"
                f"- Column: {col_s}\n"
                f"- Source: {where}\n"
                f"- Confidence: {self.confidence:.1f}\n"
                "Classify whether the stated amount is concordant with the label "
                "cell. Do NOT claim the patient received the label cell value. "
                "Do NOT replace the label cell with a neighboring cell."
            )
        return (
            f"{VERIFIED_BLOCK_MARKER}\n"
            "Your final answer MUST use this exact cell value. "
            "Do not replace it with a neighboring cell, another table, or a guessed value.\n"
            f"Your first sentence / takeaway MUST state this value verbatim: {self.value}.\n"
            "Any other conflicting numeric value in the takeaway is a critical error.\n"
            f"- Value: {self.value}\n"
            f"- Row: {row_s}\n"
            f"- Column: {col_s}\n"
            f"- Source: {where}\n"
            f"- Confidence: {self.confidence:.1f}"
        )


@dataclass
class InsufficientEvidence:
    reason: str
    structured_lookup: bool = True

    def to_prompt_block(self) -> str:
        return (
            "Insufficient table evidence:\n"
            f"- Reason: {self.reason}\n"
            "- Do not invent a cell value from flattened markdown or neighboring rows/columns.\n"
            "- If you cannot verify a unique row-column intersection, say evidence is insufficient."
        )


def _norm_unit(unit: str) -> str:
    u = (unit or "").strip().lower().replace(" ", "")
    u = u.replace("µ", "u").replace("μ", "u")
    return _UNIT_NORMALIZE.get(u, u)


def _units_compatible(a: str, b: str) -> bool:
    na, nb = _norm_unit(a), _norm_unit(b)
    if not na or not nb:
        return True
    return na == nb


def parse_range_label(text: str) -> Optional[ParsedRange]:
    raw = (text or "").strip()
    if not raw:
        return None
    m = _RANGE_RE.search(raw)
    if m:
        op, lo_s, hi_s, unit = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        lo, hi = float(lo_s), float(hi_s)
        if lo > hi:
            lo, hi = hi, lo
        inclusive_low = True
        inclusive_high = True
        if op in (">", "≥", ">="):
            inclusive_low = op in ("≥", ">=")
        elif op in ("<", "≤", "<="):
            inclusive_high = op in ("≤", "<=")
        # Patterns like ">300–400" mean (300, 400] style open on the low side.
        if raw.lstrip().startswith(">") and not raw.lstrip().startswith(">="):
            inclusive_low = False
        if raw.lstrip().startswith("≥") or raw.lstrip().startswith(">="):
            inclusive_low = True
        return ParsedRange(
            low=lo,
            high=hi,
            unit=_norm_unit(unit),
            inclusive_low=inclusive_low,
            inclusive_high=inclusive_high,
            raw=raw,
        )
    m2 = _SINGLE_BOUND_RE.search(raw)
    if m2:
        op, num_s, unit = m2.group(1), m2.group(2), m2.group(3) or ""
        num = float(num_s)
        if op in (">", "≥", ">="):
            return ParsedRange(
                low=num,
                high=None,
                unit=_norm_unit(unit),
                inclusive_low=op in ("≥", ">="),
                inclusive_high=True,
                raw=raw,
            )
        return ParsedRange(
            low=None,
            high=num,
            unit=_norm_unit(unit),
            inclusive_low=True,
            inclusive_high=op in ("≤", "<="),
            raw=raw,
        )
    return None


def range_contains(parsed: ParsedRange, value: float, unit: str = "") -> bool:
    if not _units_compatible(parsed.unit, unit):
        return False
    if parsed.low is not None:
        if parsed.inclusive_low:
            if value < parsed.low:
                return False
        elif value <= parsed.low:
            return False
    if parsed.high is not None:
        if parsed.inclusive_high:
            if value > parsed.high:
                return False
        elif value >= parsed.high:
            return False
    return True


_TIME_UNITS = frozenset({"week", "weeks", "day", "days", "month", "months", "hour", "hours", "hr", "hrs"})


def _infer_role(
    *,
    value: float,
    unit: str,
    prefix: str,
    suffix: str,
    hint: str,
) -> str:
    """Classify a numeric token without domain/product rules."""
    if _AGE_AFTER_RE.search(suffix) or hint in {"age", "aged"}:
        return ROLE_EXCLUDE
    # "4 weeks" captured with unit=weeks, or "every 4 weeks" with empty unit.
    if unit in _TIME_UNITS:
        return ROLE_FREQUENCY
    if re.search(r"(?i)\bevery\b", prefix) and (
        _FREQ_AFTER_RE.search(suffix) or unit in _TIME_UNITS
    ):
        return ROLE_FREQUENCY
    if hint in {"every", "q"} and (_FREQ_AFTER_RE.search(suffix) or unit in _TIME_UNITS):
        return ROLE_FREQUENCY
    if unit in _WEIGHT_UNITS:
        return ROLE_AXIS
    if unit in _ASSAY_UNITS:
        return ROLE_AXIS
    if unit in _MASS_UNITS and _COMPARE_HINT_RE.search(prefix):
        return ROLE_COMPARE
    if unit in _MASS_UNITS:
        # Mass amounts are usually cell values / stated amounts, not axis bands.
        return ROLE_COMPARE
    # Bare unitless numbers stay axis candidates but score lower when
    # unit-bearing axis constraints exist (blocks age→assay band matches).
    return ROLE_AXIS


def extract_numeric_constraints(question: str) -> List[NumericConstraint]:
    """Pull numeric+unit tokens from a question (domain-agnostic)."""
    q = question or ""
    found: List[NumericConstraint] = []
    seen = set()
    for m in _NUM_RE.finditer(q):
        num = float(m.group(1))
        unit = _norm_unit(m.group(2) or "")
        key = (num, unit, m.start())
        if key in seen:
            continue
        seen.add(key)
        start = max(0, m.start() - 48)
        prefix = q[start : m.start()]
        suffix = q[m.end() : m.end() + 24]
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_/-]{1,}", prefix)
        hint = tokens[-1].lower() if tokens else ""
        role = _infer_role(
            value=num, unit=unit, prefix=prefix, suffix=suffix, hint=hint
        )
        found.append(
            NumericConstraint(value=num, unit=unit, label_hint=hint, role=role)
        )
    return found


def _axis_constraints(constraints: Sequence[NumericConstraint]) -> List[NumericConstraint]:
    return [c for c in constraints if c.role == ROLE_AXIS]


def _compare_constraints(
    constraints: Sequence[NumericConstraint],
) -> List[NumericConstraint]:
    return [c for c in constraints if c.role == ROLE_COMPARE]


def _frequency_constraints(
    constraints: Sequence[NumericConstraint],
) -> List[NumericConstraint]:
    return [c for c in constraints if c.role == ROLE_FREQUENCY]


def _parse_table_payload(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
    return None


def _label_unit_cues(label: str) -> str:
    """Best-effort unit cue from a header/label (domain-agnostic)."""
    text = label or ""
    m = re.search(
        r"(?i)\b(iu/?m?l|ng/?ml|ug/?ml|µg/?ml|μg/?ml|mcg/?ml|kg|mg|g|ml|l|%)\b",
        text,
    )
    if not m:
        return ""
    return _norm_unit(m.group(1).replace("iu/ml", "iu/ml").replace("iu/mL", "iu/ml"))


def _score_label_match(
    label: str,
    constraint: NumericConstraint,
    *,
    axis_prefer_unit: str = "",
    require_unit_when_available: bool = False,
) -> float:
    if constraint.role in {ROLE_EXCLUDE, ROLE_FREQUENCY}:
        return 0.0
    # Compare-mass values must not match axis range labels.
    if constraint.role == ROLE_COMPARE:
        return 0.0

    parsed = parse_range_label(label)
    label_unit = (parsed.unit if parsed else "") or _label_unit_cues(label)
    if require_unit_when_available and constraint.unit and label_unit:
        if not _units_compatible(constraint.unit, label_unit):
            return 0.0
    # When the constraint is unitless but other axis constraints carry units,
    # heavily downrank matches against unitless ranges (blocks age→assay).
    if not constraint.unit and axis_prefer_unit and not label_unit:
        # Still allow, but weak — caller can prefer unit-bearing constraints.
        pass

    if not parsed:
        m = _NUM_RE.search(label or "")
        if not m:
            return 0.0
        if abs(float(m.group(1)) - constraint.value) > 1e-9:
            return 0.0
        if not _units_compatible(_norm_unit(m.group(2) or ""), constraint.unit):
            return 0.0
        score = 0.6
    else:
        if not range_contains(parsed, constraint.value, constraint.unit):
            return 0.0
        score = 1.0
        if constraint.label_hint and constraint.label_hint in (label or "").lower():
            score += 0.25
        if constraint.unit and _units_compatible(parsed.unit or label_unit, constraint.unit):
            score += 0.35
        elif constraint.unit and label_unit and not _units_compatible(label_unit, constraint.unit):
            return 0.0
        elif not constraint.unit:
            # Unitless constraint matching a unitless/typed range: weaker.
            score -= 0.45
        if parsed.low is not None and parsed.high is not None:
            width = max(1e-6, float(parsed.high) - float(parsed.low))
            score += min(0.5, 10.0 / width)
        elif parsed.low is not None or parsed.high is not None:
            score += 0.05

    if axis_prefer_unit and constraint.unit:
        if _units_compatible(constraint.unit, axis_prefer_unit):
            score += 0.4
        elif constraint.unit in _WEIGHT_UNITS and axis_prefer_unit in _WEIGHT_UNITS:
            score += 0.4
        elif constraint.unit in _ASSAY_UNITS and axis_prefer_unit in _ASSAY_UNITS:
            score += 0.4
    if axis_prefer_unit and not constraint.unit and axis_prefer_unit in _ASSAY_UNITS | _WEIGHT_UNITS:
        score -= 0.3
    return max(0.0, score)


def _axis_unit_preference(labels: Sequence[str]) -> str:
    units = [_label_unit_cues(x) for x in labels]
    units = [u for u in units if u]
    if not units:
        # Infer from common header words without hardcoding products.
        blob = " ".join(labels).lower()
        if re.search(r"\bkg\b|weight|body\s*mass", blob):
            return "kg"
        if re.search(r"\biu\b|assay|titer|titre|concentration|serum", blob):
            return "iu/ml"
        return ""
    # Majority unit cue.
    counts: Dict[str, int] = {}
    for u in units:
        counts[u] = counts.get(u, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _best_axis_matches(
    labels: Sequence[str],
    constraints: Sequence[NumericConstraint],
    *,
    axis_prefer_unit: str = "",
) -> List[Tuple[int, float, NumericConstraint]]:
    axis_cs = _axis_constraints(constraints)
    if not axis_cs:
        return []
    has_unit_axis = any(c.unit for c in axis_cs)
    hits: List[Tuple[int, float, NumericConstraint]] = []
    for idx, label in enumerate(labels):
        best_score = 0.0
        best_c: Optional[NumericConstraint] = None
        for c in axis_cs:
            s = _score_label_match(
                label,
                c,
                axis_prefer_unit=axis_prefer_unit,
                require_unit_when_available=has_unit_axis,
            )
            if s > best_score:
                best_score = s
                best_c = c
        if best_c is not None and best_score > 0:
            hits.append((idx, best_score, best_c))
    hits.sort(key=lambda t: t[1], reverse=True)
    return hits


def _row_frequency_text(row: Sequence[str]) -> str:
    return " ".join(str(c or "") for c in row).lower()


def _frequency_matches_row(row: Sequence[str], freq: NumericConstraint) -> bool:
    blob = _row_frequency_text(row)
    n = int(freq.value) if float(freq.value).is_integer() else freq.value
    return bool(
        re.search(
            rf"(?i)(?:every|q)\s*{n}\s*(?:weeks?|days?|months?|hours?|hrs?)\b|"
            rf"\bq{n}w\b",
            blob,
        )
    )


def lookup_cell_in_table(
    table: Dict[str, Any],
    question: str,
    *,
    filename: str = "",
    page: Optional[int] = None,
) -> VerifiedCell | InsufficientEvidence:
    """Find a unique verified cell for constraints in ``question``."""
    headers = [str(h or "") for h in (table.get("headers") or [])]
    rows = [list(r) for r in (table.get("rows") or [])]
    if not headers or not rows:
        return InsufficientEvidence("table missing headers or rows")

    width = max(len(headers), max((len(r) for r in rows), default=0))
    if width < 2:
        return InsufficientEvidence("table has fewer than 2 columns")

    headers = (headers + [""] * width)[:width]
    body = [(list(r) + [""] * width)[:width] for r in rows]
    row_labels = [str(r[0] or "") for r in body]
    col_labels = list(headers)

    constraints = extract_numeric_constraints(question)
    axis_cs = _axis_constraints(constraints)
    compare_cs = _compare_constraints(constraints)
    freq_cs = _frequency_constraints(constraints)
    if len(axis_cs) < 1:
        return InsufficientEvidence("no axis numeric constraints found in question")

    row_unit = _axis_unit_preference(row_labels + [headers[0]])
    col_unit = _axis_unit_preference(col_labels[1:])

    row_hits = _best_axis_matches(
        row_labels, axis_cs, axis_prefer_unit=row_unit
    )
    col_hits = _best_axis_matches(
        col_labels[1:], axis_cs, axis_prefer_unit=col_unit
    )
    col_hits = [(i + 1, score, c) for i, score, c in col_hits]

    if not row_hits and not col_hits:
        return InsufficientEvidence("no row or column labels matched question values")

    selected_row: Optional[Tuple[int, float, NumericConstraint]] = None
    selected_col: Optional[Tuple[int, float, NumericConstraint]] = None

    if row_hits and col_hits:
        # Prefer higher combined score with distinct constraints.
        best_pair: Optional[Tuple[float, Tuple, Tuple]] = None
        for r_idx, r_score, r_c in row_hits:
            for c_idx, c_score, c_c in col_hits:
                if r_c is c_c:
                    continue
                if (
                    abs(r_c.value - c_c.value) < 1e-9
                    and _norm_unit(r_c.unit) == _norm_unit(c_c.unit)
                    and _norm_unit(r_c.unit)
                ):
                    continue
                # Prefer unit-aligned axes (assay→rows, mass-of-patient→cols).
                bonus = 0.0
                if r_c.unit and row_unit and _units_compatible(r_c.unit, row_unit):
                    bonus += 0.2
                if c_c.unit and col_unit and _units_compatible(c_c.unit, col_unit):
                    bonus += 0.2
                if r_c.unit in _WEIGHT_UNITS and c_c.unit in _ASSAY_UNITS:
                    # Likely swapped axes — downrank.
                    bonus -= 0.5
                total = r_score + c_score + bonus
                pair = (total, (r_idx, r_score, r_c), (c_idx, c_score, c_c))
                if best_pair is None or total > best_pair[0]:
                    best_pair = pair
        if best_pair is not None:
            selected_row = best_pair[1]  # type: ignore[assignment]
            selected_col = best_pair[2]  # type: ignore[assignment]
    elif row_hits and len(axis_cs) >= 2 and width >= 3:
        primary_constraint = row_hits[0][2]
        candidate_rows = [
            (idx, score, c)
            for idx, score, c in row_hits
            if c is primary_constraint
            or (
                abs(c.value - primary_constraint.value) < 1e-9
                and _norm_unit(c.unit) == _norm_unit(primary_constraint.unit)
            )
        ]
        if not candidate_rows:
            candidate_rows = [row_hits[0]]
        remaining = [
            c
            for c in axis_cs
            if not (
                abs(c.value - primary_constraint.value) < 1e-9
                and _norm_unit(c.unit) == _norm_unit(primary_constraint.unit)
            )
        ]
        if not remaining:
            return InsufficientEvidence("need a second distinct constraint for list lookup")

        verified_hits: List[VerifiedCell] = []
        for r_idx, r_score, _r_c in candidate_rows:
            for c in remaining:
                for col_i in range(1, width - 1):
                    cell_label = str(body[r_idx][col_i] or "")
                    if _score_label_match(cell_label, c) <= 0:
                        continue
                    value = str(body[r_idx][width - 1] or "").strip()
                    if not value:
                        continue
                    verified_hits.append(
                        VerifiedCell(
                            value=value,
                            row_headers=[row_labels[r_idx], cell_label],
                            col_headers=[headers[width - 1]],
                            row_index=r_idx,
                            col_index=width - 1,
                            page=page if page is not None else table.get("page"),
                            filename=filename,
                            confidence=min(100.0, 70.0 + 15.0 * r_score),
                            table_headers=headers,
                            evidence={
                                "match_mode": "list_row_with_mid_range",
                                "mid_col": col_i,
                                "constraints": [
                                    {
                                        "value": primary_constraint.value,
                                        "unit": primary_constraint.unit,
                                    },
                                    {"value": c.value, "unit": c.unit},
                                ],
                            },
                        )
                    )
        if len(verified_hits) == 1:
            cell = verified_hits[0]
            return _attach_compare(cell, compare_cs)
        if len(verified_hits) > 1:
            values = {h.value for h in verified_hits}
            if len(values) == 1:
                return _attach_compare(verified_hits[0], compare_cs)
            return InsufficientEvidence(
                "multiple list rows match with disagreeing values"
            )
        return InsufficientEvidence("could not uniquely match secondary range in row")
    elif row_hits and len(col_labels) == 2:
        selected_row = row_hits[0]
        selected_col = (1, 1.0, selected_row[2])
    elif col_hits and len(row_labels) == 1:
        selected_col = col_hits[0]
        selected_row = (0, 1.0, selected_col[2])

    if not selected_row or not selected_col:
        if row_hits and col_hits:
            return InsufficientEvidence(
                "ambiguous row/column pairing for question constraints"
            )
        return InsufficientEvidence("unable to resolve both row and column axes")

    r_i = selected_row[0]
    c_i = selected_col[0]

    # Frequency disambiguation when several rows share the same axis band.
    # Skip frequency tokens tied to a compare/prescribed amount — those describe
    # what was given, not which label row to select.
    standalone_freq = []
    for f in freq_cs:
        pattern = (
            rf"(?i)(?:prescrib\w*|receiv\w*|given|administer\w*|order(?:ed)?|"
            rf"dispens\w*).{{0,40}}{int(f.value)}\s*(?:weeks?|days?|months?|hours?)"
        )
        if re.search(pattern, question or ""):
            continue
        standalone_freq.append(f)
    if standalone_freq:
        same_band = [
            (idx, score, c)
            for idx, score, c in row_hits
            if c is selected_row[2]
            or (
                abs(c.value - selected_row[2].value) < 1e-9
                and _norm_unit(c.unit) == _norm_unit(selected_row[2].unit)
            )
        ]
        freq_matched = [
            item
            for item in same_band
            if _frequency_matches_row(body[item[0]], standalone_freq[0])
        ]
        if len(freq_matched) == 1:
            r_i = freq_matched[0][0]
            selected_row = freq_matched[0]

    if r_i < 0 or r_i >= len(body) or c_i < 0 or c_i >= width:
        return InsufficientEvidence("resolved indexes out of range")

    value = str(body[r_i][c_i] or "").strip()
    if not value:
        return InsufficientEvidence("intersecting cell is empty")

    if _score_label_match(row_labels[r_i], selected_row[2], axis_prefer_unit=row_unit) <= 0:
        return InsufficientEvidence("row label failed verification against constraint")
    if c_i > 0 and _score_label_match(
        col_labels[c_i], selected_col[2], axis_prefer_unit=col_unit
    ) <= 0:
        if selected_col[2] is selected_row[2] and len(axis_cs) == 1:
            pass
        elif not parse_range_label(col_labels[c_i]):
            pass
        else:
            return InsufficientEvidence(
                "column label failed verification against constraint"
            )

    conflicts = []
    for other_i, other_score, other_c in row_hits:
        if other_i == r_i:
            continue
        if other_c is not selected_row[2] and not (
            abs(other_c.value - selected_row[2].value) < 1e-9
            and _norm_unit(other_c.unit) == _norm_unit(selected_row[2].unit)
        ):
            continue
        if abs(other_score - selected_row[1]) < 1e-9:
            # If standalone frequency uniquely selects our row, ignore siblings.
            if standalone_freq and _frequency_matches_row(
                body[r_i], standalone_freq[0]
            ):
                if not _frequency_matches_row(body[other_i], standalone_freq[0]):
                    continue
            other_val = str(body[other_i][c_i] or "").strip()
            if other_val and other_val != value:
                conflicts.append(other_val)
    if conflicts:
        return InsufficientEvidence(
            "multiple rows match equally with disagreeing cell values"
        )

    cell = VerifiedCell(
        value=value,
        row_headers=[row_labels[r_i]],
        col_headers=[col_labels[c_i]],
        row_index=r_i,
        col_index=c_i,
        page=page if page is not None else table.get("page"),
        filename=filename,
        confidence=min(
            100.0,
            60.0 + 20.0 * float(selected_row[1]) + 10.0 * float(selected_col[1]),
        ),
        table_headers=headers,
        evidence={
            "match_mode": "row_col_intersection",
            "constraints": [
                {
                    "value": selected_row[2].value,
                    "unit": selected_row[2].unit,
                    "role": selected_row[2].role,
                },
                {
                    "value": selected_col[2].value,
                    "unit": selected_col[2].unit,
                    "role": selected_col[2].role,
                },
            ],
        },
    )
    return _attach_compare(cell, compare_cs)


def _attach_compare(
    cell: VerifiedCell,
    compare_cs: Sequence[NumericConstraint],
) -> VerifiedCell:
    if not compare_cs:
        return cell
    # Prefer the first mass/compare token (usually the stated administered amount).
    c = compare_cs[0]
    evidence = dict(cell.evidence or {})
    evidence["match_mode"] = (
        "axis_lookup_with_prescribed_compare"
        if evidence.get("match_mode") == "row_col_intersection"
        else evidence.get("match_mode") or "axis_lookup_with_prescribed_compare"
    )
    evidence["prescribed_value"] = c.value
    evidence["prescribed_unit"] = c.unit
    cell.evidence = evidence
    return cell


def lookup_from_chunk(
    chunk: Dict[str, Any],
    question: str,
) -> VerifiedCell | InsufficientEvidence | None:
    table = _parse_table_payload(chunk.get("table_json"))
    if not table:
        return None
    page = chunk.get("page")
    try:
        page_i = int(page) if page is not None else table.get("page")
    except (TypeError, ValueError):
        page_i = table.get("page")
    return lookup_cell_in_table(
        table,
        question,
        filename=str(chunk.get("filename") or chunk.get("document_name") or ""),
        page=page_i if isinstance(page_i, int) else None,
    )
