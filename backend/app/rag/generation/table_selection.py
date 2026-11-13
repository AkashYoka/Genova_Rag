"""Domain-agnostic table eligibility and multi-table conflict detection.

Compares query constraints to table applicability metadata without hardcoded
domain schemas. Sibling tables that share numeric axes but differ in
applicability must not be silently collapsed to one answer.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.rag.generation.query_constraints import (
    QueryConstraints,
    extract_query_constraints,
    mark_missing_dimensions,
)
from app.rag.generation.structured_lookup import (
    ROLE_AXIS,
    ROLE_EXCLUDE,
    ParsedRange,
    extract_numeric_constraints,
    parse_range_label,
    range_contains,
)
from app.rag.ingestion.pdf_layout.table_structure import (
    STRUCTURE_INVALID,
    STRUCTURE_NEEDS_VERIFICATION,
    validate_table_structure,
)

# Calendar-duration units used as applicability bounds (not domain-specific ages).
_DURATION_UNIT_HINTS = frozenset({"year", "years", "yo", "yr", "yrs"})

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[/-][a-z0-9]+)?", re.I)
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "from",
        "by",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "these",
        "those",
        "table",
        "tables",
        "page",
        "see",
        "per",
        "vs",
        "versus",
        # Generic lookup vocabulary — must not uniquely select a table.
        "value",
        "values",
        "amount",
        "amounts",
        "recommended",
        "every",
        "weeks",
        "week",
        "what",
        "which",
        "according",
        "lookup",
        "matrix",
        "grid",
        "chart",
        "schedule",
        "parameter",
        "parameters",
        "row",
        "column",
        "cell",
        "dose",
        "doses",
        "dosing",
        "dosage",
        "freq",
        "frequency",
        "strength",
        "presentation",
        "body",
        "weight",
        "mass",
        "load",
        "stress",
        "margin",
        "revenue",
        "response",
        "concentration",
    }
)


@dataclass
class TableApplicability:
    table_id: str = ""
    cues: List[str] = field(default_factory=list)
    cue_tokens: Set[str] = field(default_factory=set)
    attributes: Dict[str, str] = field(default_factory=dict)
    # Generic unit-tagged applicability ranges (years, etc.) parsed from cues.
    bound_ranges: List[ParsedRange] = field(default_factory=list)
    differentiating_dimensions: List[str] = field(default_factory=list)
    header_units: Set[str] = field(default_factory=set)
    purpose_tokens: Set[str] = field(default_factory=set)
    blob: str = ""
    # Back-compat alias used by older tests.
    age_ranges: List[ParsedRange] = field(default_factory=list)


ELIGIBLE = "eligible"
INELIGIBLE = "ineligible"
UNCERTAIN = "uncertain"

_POPULATION_KEY_HINTS = (
    "population",
    "cohort",
    "age",
    "subject",
    "patient",
    "audience",
    "segment",
    "group",
)


@dataclass
class CandidateEligibility:
    """Explicit eligibility state for one candidate table (audit record)."""

    table_id: str = ""
    state: str = UNCERTAIN
    reasons: List[str] = field(default_factory=list)
    question_answered: str = ""
    population: str = ""
    dimensions: Dict[str, List[str]] = field(default_factory=dict)
    units: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    sufficient_evidence: bool = False
    provenance: Dict[str, str] = field(default_factory=dict)
    applicability: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _applicability_state(
    table: Dict[str, Any],
    profile: "TableApplicability",
) -> Tuple[str, bool, List[str]]:
    """ELIGIBLE only when applicability evidence exists; else UNCERTAIN.

    Provenance-only attributes (section/caption/trial) do not establish
    applicability, so they never promote a candidate to ELIGIBLE.
    """
    und = _understanding(table)
    app_attrs = {
        k: v
        for k, v in (profile.attributes or {}).items()
        if dimension_kind(k) == "applicability"
    }
    has_bounds = bool(profile.bound_ranges or profile.age_ranges)
    reasons: List[str] = []
    if und.get("uncertain") or und.get("source") == "deterministic_fallback":
        reasons.append("interpretation_uncertain")
    if app_attrs or has_bounds:
        return ELIGIBLE, True, reasons
    if profile.cue_tokens:
        reasons.append("metadata_without_applicability_attributes")
        return UNCERTAIN, False, reasons
    reasons.append("no_applicability_metadata")
    return UNCERTAIN, False, reasons


def _eligibility_records(
    profiles: Sequence[Tuple[Dict[str, Any], Dict[str, Any], "TableApplicability"]],
    eligible: Sequence[Tuple[Dict[str, Any], Dict[str, Any], "TableApplicability"]],
    excluded: Sequence[Dict[str, Any]],
) -> List[CandidateEligibility]:
    """Per-candidate eligibility states for audit (no selection side effects)."""
    eligible_ids = {
        str(profile.table_id or table.get("table_id") or "")
        for _chunk, table, profile in eligible
    }
    excluded_reasons: Dict[str, str] = {}
    for item in excluded or []:
        tid = str(item.get("table_id") or "")
        if tid and tid not in excluded_reasons:
            excluded_reasons[tid] = str(item.get("reason") or "excluded")

    profiles_by_id: Dict[str, "TableApplicability"] = {}
    for chunk, table, profile in profiles:
        tid = str(profile.table_id or table.get("table_id") or chunk.get("table_id") or "")
        if tid:
            profiles_by_id[tid] = profile

    records: List[CandidateEligibility] = []
    for chunk, table, profile in profiles:
        tid = str(profile.table_id or table.get("table_id") or chunk.get("table_id") or "")
        und = _understanding(table)
        provenance = {
            "caption": str(table.get("caption") or chunk.get("caption") or ""),
            "section": str(chunk.get("section") or table.get("section") or ""),
            "page": str(table.get("page") or chunk.get("page") or ""),
            "table_id": tid,
            "document_id": str(chunk.get("document_id") or ""),
        }
        population = ""
        for key, val in (profile.attributes or {}).items():
            if any(hint in key.lower() for hint in _POPULATION_KEY_HINTS):
                population = str(val)
                break
        dimensions: Dict[str, List[str]] = {}
        for key, val in (profile.attributes or {}).items():
            dimensions.setdefault(key, []).append(str(val))
        record = CandidateEligibility(
            table_id=tid,
            question_answered=str(
                und.get("purpose") or und.get("subject") or provenance["caption"]
            ),
            population=population,
            dimensions=dimensions,
            units=sorted(profile.header_units or set()),
            provenance=provenance,
            applicability=dict(profile.attributes or {}),
        )
        if tid in eligible_ids:
            state, sufficient, reasons = _applicability_state(table, profile)
            record.state = state
            record.sufficient_evidence = sufficient
            record.reasons = reasons
        else:
            record.state = INELIGIBLE
            record.reasons = [
                excluded_reasons.get(tid, "not_selected_by_query_applicability")
            ]
        records.append(record)

    applicable = [r for r in records if r.state in {ELIGIBLE, UNCERTAIN}]
    for i, left in enumerate(applicable):
        for right in applicable[i + 1 :]:
            left_profile = profiles_by_id.get(left.table_id)
            right_profile = profiles_by_id.get(right.table_id)
            if left_profile is None or right_profile is None:
                continue
            if profiles_are_distinct(left_profile, right_profile):
                left.conflicts_with.append(right.table_id)
                right.conflicts_with.append(left.table_id)
    return records


def eligibility_audit(decision: "SelectionDecision") -> List[Dict[str, Any]]:
    """Serializable eligibility decisions for telemetry/debugging."""
    return [c.to_dict() for c in (decision.candidates or [])]


@dataclass
class SelectionDecision:
    """Result of filtering candidate table chunks for structured lookup."""

    eligible_chunks: List[Dict[str, Any]]
    excluded: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    requires_disambiguation: bool = False
    missing_dimensions: List[str] = field(default_factory=list)
    query_constraints: Optional[Dict[str, Any]] = None
    structure_blocked: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[CandidateEligibility] = field(default_factory=list)
    # Condition words from the question that no candidate table covers.
    mismatch_terms: List[str] = field(default_factory=list)


def attribute_value_tokens(profile: TableApplicability) -> Set[str]:
    """Token set of all declared attribute values (domain-agnostic)."""
    toks: Set[str] = set()
    for val in (profile.attributes or {}).values():
        toks |= _tokens(str(val))
    return toks


def profile_query_attr_hits(
    profile: TableApplicability,
    q_tokens: Set[str],
) -> Set[str]:
    """Attribute-value tokens from ``profile`` that also appear in the query."""
    if not q_tokens:
        return set()
    return attribute_value_tokens(profile) & q_tokens


def _tokens(text: str) -> Set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) > 2 and t.lower() not in _STOP
    }


# Cohort words describe which table the question names, not which indication
# the patient has; they are ignored when choosing a replacement table.
_COHORT_TOKENS = frozenset(
    {
        "adult",
        "adults",
        "adolescent",
        "adolescents",
        "pediatric",
        "paediatric",
        "child",
        "children",
        "infant",
        "infants",
    }
)


_GENERIC_QUERY_TOKENS = frozenset(
    {
        "patient",
        "patients",
        "adult",
        "adults",
        "pediatric",
        "paediatric",
        "child",
        "children",
        "label",
        "correct",
        "appropriate",
        "given",
        "dosed",
        "using",
        "regimen",
        "table",
        "tables",
        "value",
        "values",
        "dose",
        "doses",
        "dosing",
        "frequency",
        "recommended",
        "recommend",
        "according",
        "received",
        "weighs",
    }
)

# "... with <condition phrase>" — the phrase, not the whole sentence, names
# the patient's condition.
_CONDITION_PHRASE_RE = re.compile(
    r"(?i)\b(?:with|diagnosed\s+with|history\s+of)\s+"
    r"([a-z][a-z\s\-()]{2,50}?)"
    r"(?=[,.;:]|$|\s+\d|\s+(?:weigh\w*|body|was|were|is|received|given|got|"
    r"dosed|using|at|who)\b)"
)

# Measurement nouns that can follow "with" but do not name a condition.
_NON_CONDITION_TOKENS = frozenset(
    {"assay", "level", "levels", "result", "results", "count", "score", "ratio"}
)


def _condition_tokens(question: str) -> Set[str]:
    """Content tokens of the condition phrase(s) stated in the question."""
    toks: Set[str] = set()
    for m in _CONDITION_PHRASE_RE.finditer(question or ""):
        toks |= _tokens(m.group(1))
    return toks - _GENERIC_QUERY_TOKENS - _NON_CONDITION_TOKENS


def _indication_mismatch_terms(
    question: str,
    eligible: Sequence[Tuple[Dict[str, Any], Dict[str, Any], "TableApplicability"]],
) -> List[str]:
    """Condition words the question states that no candidate table covers."""
    condition = _condition_tokens(question)
    if not condition or not eligible:
        return []
    covered: Set[str] = set()
    for _chunk, _table, profile in eligible:
        covered |= set(profile.cue_tokens or set()) | set(
            profile.purpose_tokens or set()
        )
    uncovered = sorted(condition - covered)
    # One stray word can be a qualifier; several mean the condition itself is
    # outside every candidate table's scope.
    return uncovered if len(uncovered) >= 2 else []


def _narrow_by_purpose_overlap(
    eligible: Sequence[Tuple[Dict[str, Any], Dict[str, Any], "TableApplicability"]],
    q_tokens: Set[str],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], "TableApplicability"]]:
    """Pick the single surviving table whose purpose best matches the question."""
    if len(eligible) < 2:
        return list(eligible)
    probe = set(q_tokens) - _COHORT_TOKENS
    scored = [
        (len(set(item[2].purpose_tokens or set()) & probe), item)
        for item in eligible
    ]
    best = max(score for score, _ in scored)
    if best <= 0:
        return list(eligible)
    winners = [item for score, item in scored if score == best]
    return winners if len(winners) == 1 else list(eligible)


def _understanding(table: Dict[str, Any]) -> Dict[str, Any]:
    raw = table.get("llm_understanding")
    return raw if isinstance(raw, dict) else {}


def _header_units(table: Dict[str, Any]) -> Set[str]:
    units: Set[str] = set()
    blobs = list(table.get("headers") or [])
    for path in table.get("normalized_header_paths") or []:
        blobs.extend(str(p) for p in (path or []))
    for row in (table.get("rows") or [])[:8]:
        if row:
            blobs.append(str(row[0] or ""))
    for text in blobs:
        for m in re.finditer(
            r"(?i)\b(kg|mg|g|%|iu/?ml|ng/?ml|mpa|psi|mm|cm|m|t|ton|usd|eur|fy)\b",
            text or "",
        ):
            units.add(m.group(1).lower().replace(" ", ""))
    return units


def build_table_applicability(
    table: Dict[str, Any],
    *,
    chunk: Optional[Dict[str, Any]] = None,
) -> TableApplicability:
    """Collect applicability cues from interpretation + caption/section text."""
    chunk = chunk or {}
    und = _understanding(table)
    cues: List[str] = []
    for item in und.get("applicability_cues") or []:
        if str(item or "").strip():
            cues.append(str(item).strip())
    attrs: Dict[str, str] = {}
    for item in und.get("applicability_attributes") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if key and value:
            attrs[key.lower()] = value
            cues.append(f"{key}: {value}")
    subject = str(und.get("subject") or "").strip()
    purpose = str(und.get("purpose") or "").strip()
    caption = str(table.get("caption") or chunk.get("caption") or "").strip()
    section = " ".join(
        str(x or "")
        for x in (
            chunk.get("section"),
            chunk.get("subsection"),
            chunk.get("section_path"),
        )
    ).strip()
    for phrase in und.get("semantic_search_phrases") or []:
        if str(phrase or "").strip():
            cues.append(str(phrase).strip())
    for extra in (subject, purpose, caption, section):
        if extra:
            cues.append(extra)

    blob = " | ".join(dict.fromkeys(cues))
    bound_ranges: List[ParsedRange] = []
    for cue in cues:
        bound_ranges.extend(_extract_bound_ranges_from_text(cue))

    dims = [
        str(d).strip()
        for d in (und.get("differentiating_dimensions") or [])
        if str(d or "").strip()
    ]
    # Promote attribute keys that siblings may conflict on.
    for key in attrs:
        if key not in {d.lower() for d in dims}:
            dims.append(key)

    table_id = str(
        table.get("table_id") or chunk.get("table_id") or und.get("table_id") or ""
    )
    purpose_tokens = _tokens(" ".join(x for x in (purpose, subject, caption) if x))
    return TableApplicability(
        table_id=table_id,
        cues=list(dict.fromkeys(cues)),
        cue_tokens=_tokens(blob),
        attributes=attrs,
        bound_ranges=bound_ranges,
        age_ranges=list(bound_ranges),  # back-compat
        differentiating_dimensions=list(dict.fromkeys(dims)),
        header_units=_header_units(table),
        purpose_tokens=purpose_tokens,
        blob=blob,
    )


_TABLE_NUM_RE = re.compile(r"(?i)\btable\s+(\d+)\b")


def question_table_number(question: str) -> Optional[str]:
    """Return explicit ``Table N`` cited in the question, if any."""
    m = _TABLE_NUM_RE.search(question or "")
    return m.group(1) if m else None


def table_number_match(
    table: Dict[str, Any],
    chunk: Optional[Dict[str, Any]],
    num: str,
) -> bool:
    """True when caption/section/headers mention the same Table N."""
    chunk = chunk or {}
    blobs = [
        str(table.get("caption") or ""),
        str(chunk.get("caption") or ""),
        str(chunk.get("section") or ""),
        str(table.get("section") or ""),
    ]
    for h in table.get("headers") or []:
        blobs.append(str(h or ""))
    blob = " ".join(blobs)
    return bool(re.search(rf"(?i)\btable\s+{re.escape(str(num))}\b", blob))


def _extract_bound_ranges_from_text(cue: str) -> List[ParsedRange]:
    """Parse unit-tagged applicability intervals from free text (e.g. years)."""
    ranges: List[ParsedRange] = []
    if not cue:
        return ranges
    for m in re.finditer(
        r"(?i)(?:ages?\s*(?:of\s+)?)?(≥|>=|>|≤|<=|<)?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:–|-|—|to)\s*"
        r"(≥|>=|>|≤|<=|<)?\s*(\d+(?:\.\d+)?)\s*years?",
        cue,
    ):
        low = float(m.group(2))
        high = float(m.group(4))
        op_low = (m.group(1) or "").lower()
        op_high = (m.group(3) or "").lower()
        inclusive_low = op_low not in {">"}
        inclusive_high = op_high not in {"<"} if op_high else True
        if op_high in {"<", "≤", "<="}:
            inclusive_high = op_high in {"≤", "<="}
        ranges.append(
            ParsedRange(
                low=low,
                high=high,
                unit="years",
                inclusive_low=inclusive_low,
                inclusive_high=inclusive_high,
                raw=m.group(0),
            )
        )
    for m in re.finditer(
        r"(?i)(≥|>=|>|≤|<=|<)\s*(\d+(?:\.\d+)?)\s*years?",
        cue,
    ):
        bound = float(m.group(2))
        op = (m.group(1) or "").lower()
        if op in {">", ">=", "≥"}:
            ranges.append(
                ParsedRange(
                    low=bound,
                    high=None,
                    unit="years",
                    inclusive_low=op in {">=", "≥"},
                    inclusive_high=True,
                    raw=m.group(0),
                )
            )
        else:
            ranges.append(
                ParsedRange(
                    low=None,
                    high=bound,
                    unit="years",
                    inclusive_low=True,
                    inclusive_high=op in {"<=", "≤"},
                    raw=m.group(0),
                )
            )
    for m in re.finditer(
        r"(?i)(\d+(?:\.\d+)?)\s*years?(?:\s+of\s+age)?\s+"
        r"(?:and\s+)?(?:older|above|over)",
        cue,
    ):
        bound = float(m.group(1))
        ranges.append(
            ParsedRange(
                low=bound,
                high=None,
                unit="years",
                inclusive_low=True,
                inclusive_high=True,
                raw=m.group(0),
            )
        )
    for m in re.finditer(
        r"(?i)(?:patients?|persons?|subjects?|adults?)\s+"
        r"(\d+(?:\.\d+)?)\s*years?\s+of(?:\s+age)?\b",
        cue,
    ):
        if re.search(r"(?i)\b(?:older|above|over)\b", cue):
            bound = float(m.group(1))
            ranges.append(
                ParsedRange(
                    low=bound,
                    high=None,
                    unit="years",
                    inclusive_low=True,
                    inclusive_high=True,
                    raw=m.group(0),
                )
            )
    parsed = parse_range_label(cue)
    if parsed and ("year" in cue.lower()):
        ranges.append(parsed)
    return ranges


_COHORT_YEAR_PHRASE_RE = re.compile(
    r"(?i)(?:>=?\s*|≤\s*|<=\s*|>\s*|<\s*)?(\d+(?:\.\d+)?)\s*"
    r"(?:years?(?:\s+of\s+age)?|yrs?|yo)\b"
    r"(?:\s+and\s+older|\s+and\s+above|\s+or\s+older)?"
)


def question_age_constraints(question: str) -> List[float]:
    """Calendar-duration cohort numerics from the question (domain-agnostic).

    Includes classic ``N-year-old`` ROLE_EXCLUDE tokens and phrases such as
    ``>=12 years``, ``12 years and older`` that extractors may label as axis.
    """
    bounds: List[float] = []
    q = question or ""
    for c in extract_numeric_constraints(q):
        if c.role == ROLE_EXCLUDE:
            pattern = (
                rf"(?i){re.escape(str(int(c.value) if float(c.value).is_integer() else c.value))}"
                rf"\s*-?\s*(?:year|yr|yo)"
            )
            if re.search(pattern, q) or (c.unit or "").lower() in _DURATION_UNIT_HINTS:
                bounds.append(float(c.value))
                continue
        # Duration-unit numerics used as cohort bounds (not matrix axes).
        if (c.unit or "").lower() in _DURATION_UNIT_HINTS and re.search(
            rf"(?i){re.escape(str(int(c.value) if float(c.value).is_integer() else c.value))}"
            rf".{{0,12}}(?:year|yr|yo).{{0,16}}(?:older|above|younger|under|age)",
            q,
        ):
            bounds.append(float(c.value))
    for m in _COHORT_YEAR_PHRASE_RE.finditer(q):
        try:
            bounds.append(float(m.group(1)))
        except ValueError:
            continue
    # Deduplicate while preserving order.
    out: List[float] = []
    seen: set = set()
    for b in bounds:
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def _bound_compatible(value: float, ranges: Sequence[ParsedRange]) -> Optional[bool]:
    if not ranges:
        return None
    return any(range_contains(r, value, unit=r.unit or "years") for r in ranges)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _normalize_bound_unit(unit: str) -> str:
    u = (unit or "").strip().lower().replace(" ", "")
    aliases = {
        "yr": "years",
        "yrs": "years",
        "year": "years",
        "yo": "years",
        "y": "years",
    }
    return aliases.get(u, u)


def _bound_units_compatible(left: str, right: str) -> bool:
    a = _normalize_bound_unit(left)
    b = _normalize_bound_unit(right)
    if not a or not b:
        # Unknown unit on either side → comparable only when both unknown.
        return not a and not b
    return a == b


def _range_endpoints(
    parsed: ParsedRange,
) -> Tuple[Optional[float], Optional[float], bool, bool]:
    """Return (low, high, inclusive_low, inclusive_high) with open ends as None."""
    return (
        parsed.low,
        parsed.high,
        bool(parsed.inclusive_low),
        bool(parsed.inclusive_high),
    )


def ranges_are_equivalent(left: ParsedRange, right: ParsedRange) -> bool:
    """True when two bounds describe the same interval (unit + endpoints)."""
    if not _bound_units_compatible(left.unit, right.unit):
        return False
    return _range_endpoints(left) == _range_endpoints(right)


def ranges_overlap(left: ParsedRange, right: ParsedRange) -> bool:
    """Proper interval overlap respecting inclusive/exclusive bounds.

    Different / unknown-incompatible units never overlap.
    """
    if not _bound_units_compatible(left.unit, right.unit):
        return False
    # Disjoint if left entirely below right.
    if left.high is not None and right.low is not None:
        if left.high < right.low:
            return False
        if left.high == right.low and not (
            left.inclusive_high and right.inclusive_low
        ):
            return False
    # Disjoint if right entirely below left.
    if right.high is not None and left.low is not None:
        if right.high < left.low:
            return False
        if right.high == left.low and not (
            right.inclusive_high and left.inclusive_low
        ):
            return False
    return True


def bound_sets_are_distinct(
    left_ranges: Sequence[ParsedRange],
    right_ranges: Sequence[ParsedRange],
) -> bool:
    """True when bound sets represent different applicability classes.

    - No compatible-unit pair → distinct (incomparable units / empty).
    - Any equivalent interval pair → not distinct on bounds alone.
    - Partial overlap or containment with unequal endpoints → distinct.
    - Fully disjoint compatible ranges → distinct.
    """
    if not left_ranges or not right_ranges:
        return False
    comparable = False
    any_equivalent = False
    any_partial_or_disjoint = False
    for a in left_ranges:
        for b in right_ranges:
            if not _bound_units_compatible(a.unit, b.unit):
                continue
            comparable = True
            if ranges_are_equivalent(a, b):
                any_equivalent = True
                continue
            # Unequal intervals: overlapping or disjoint both mean different classes.
            any_partial_or_disjoint = True
    if not comparable:
        # Units never aligned — treat as distinct applicability conditions.
        return True
    if any_equivalent and not any_partial_or_disjoint:
        return False
    return any_partial_or_disjoint or not any_equivalent


def profiles_are_distinct(
    left: TableApplicability,
    right: TableApplicability,
    *,
    max_overlap: float = 0.35,
) -> bool:
    """True when two tables look like different applicability classes."""
    left_ranges = left.bound_ranges or left.age_ranges
    right_ranges = right.bound_ranges or right.age_ranges
    if left_ranges and right_ranges and bound_sets_are_distinct(left_ranges, right_ranges):
        return True
    # One side declares bounds and the other does not → different classes when
    # cue tokens also diverge (missing metadata is not a wildcard).
    if bool(left_ranges) != bool(right_ranges):
        if left.cue_tokens and right.cue_tokens:
            if jaccard(left.cue_tokens, right.cue_tokens) <= 0.55:
                return True
        elif left.attributes or right.attributes:
            return True
    overlap = jaccard(left.cue_tokens, right.cue_tokens)
    if overlap <= max_overlap and (left.cue_tokens or right.cue_tokens):
        return True
    shared_keys = {
        k
        for k in (set(left.attributes) & set(right.attributes))
        if dimension_kind(k) != "provenance"
    }
    for key in shared_keys:
        if left.attributes[key].lower() != right.attributes[key].lower():
            return True
    # Asymmetric applicability attributes with non-trivial unique tokens.
    only_left = {
        k
        for k in (set(left.attributes) - set(right.attributes))
        if dimension_kind(k) != "provenance"
    }
    only_right = {
        k
        for k in (set(right.attributes) - set(left.attributes))
        if dimension_kind(k) != "provenance"
    }
    if (only_left or only_right) and (left.attributes or right.attributes):
        if left.cue_tokens and right.cue_tokens and overlap <= 0.6:
            return True
    return False


# Provenance / document-layout dims are not patient-applicability constraints.
# They must not alone force hard disambiguation of dosing/lookup matrices.
_PROVENANCE_DIMENSIONS = frozenset(
    {
        "section",
        "subsection",
        "section_path",
        "caption",
        "title",
        "trial_name",
        "trial",
        "study",
        "study_name",
        "document",
        "source",
        "page",
        "pages",
        "filename",
        "extraction_method",
        "provenance",
        "applicability_class",  # generic placeholder, not a real split key
    }
)


def dimension_kind(dim: str) -> str:
    """Classify a dimension as applicability vs provenance vs other."""
    key = str(dim or "").strip().lower().replace(" ", "_")
    if not key:
        return "other"
    if key in _PROVENANCE_DIMENSIONS or key.startswith("trial_") or key.startswith(
        "study_"
    ):
        return "provenance"
    if "section" in key or key in {"caption", "title"}:
        return "provenance"
    return "applicability"


def applicability_dimensions(dims: Sequence[str]) -> List[str]:
    return [d for d in dims if dimension_kind(d) == "applicability"]


def conflicting_dimensions(
    left: TableApplicability,
    right: TableApplicability,
) -> List[str]:
    """Return attribute keys / dims that distinguish two profiles.

    Only dims that actually differ between the two profiles are returned.
    Declared ``differentiating_dimensions`` are included only when they name a
    shared attribute with unequal values, a one-sided attribute key, or a
    bound conflict — not merely because the LLM listed them on one table.
    """
    dims: List[str] = []
    shared_keys = set(left.attributes) & set(right.attributes)
    for key in shared_keys:
        if dimension_kind(key) == "provenance":
            continue
        if left.attributes[key].lower() != right.attributes[key].lower():
            dims.append(key)
    left_only = set(left.attributes) - set(right.attributes)
    right_only = set(right.attributes) - set(left.attributes)
    # Unique attribute keys are differentiating when both sides have attrs.
    if left.attributes and right.attributes:
        dims.extend(
            sorted(
                k
                for k in (left_only | right_only)
                if dimension_kind(k) != "provenance"
            )
        )
    left_ranges = left.bound_ranges or left.age_ranges
    right_ranges = right.bound_ranges or right.age_ranges
    if left_ranges and right_ranges and bound_sets_are_distinct(left_ranges, right_ranges):
        dims.append("applicability_bound")
    declared = list(
        dict.fromkeys(
            list(left.differentiating_dimensions or [])
            + list(right.differentiating_dimensions or [])
        )
    )
    for d in declared:
        key = str(d or "").strip()
        if not key or key in dims:
            continue
        # Keep declared dim only when it corresponds to a real attribute split
        # or bound conflict already detected above (via alias match).
        lk = key.lower()
        if lk in {k.lower() for k in dims}:
            continue
        # Skip provenance-only declarations — they are not eligibility splits.
        if dimension_kind(key) == "provenance":
            continue
        # If neither profile has this attribute key, do not invent a conflict.
        left_val = next(
            (v for k, v in (left.attributes or {}).items() if k.lower() == lk), None
        )
        right_val = next(
            (v for k, v in (right.attributes or {}).items() if k.lower() == lk), None
        )
        if left_val is None and right_val is None:
            continue
        # Both sides declare the same value: metadata agreement, not a split.
        if (
            left_val is not None
            and right_val is not None
            and left_val.strip().lower() == right_val.strip().lower()
        ):
            continue
        dims.append(key)
    return list(dict.fromkeys(dims))


def _query_units(constraints: QueryConstraints) -> Set[str]:
    units: Set[str] = set()
    for c in constraints.constraints:
        if c.unit:
            units.add(c.unit.lower().replace(" ", ""))
        if c.role == ROLE_AXIS and c.unit:
            units.add(c.unit.lower().replace(" ", ""))
    return units


def _axis_compatible(table: Dict[str, Any], constraints: QueryConstraints) -> bool:
    """Reject tables whose headers cannot host the query's unit-bearing axes.

    Multi-axis numeric questions (e.g. two unit-bearing factors) require the
    table to expose matching header units. Unit-less siblings (adverse-event
    lists, narrative grids) must not enter the eligibility pool and create
    false applicability conflicts with the real matrix.
    """
    axis_units = {
        (c.unit or "").lower().replace(" ", "")
        for c in constraints.constraints
        if c.role == ROLE_AXIS and c.unit
    }
    if len(axis_units) < 1:
        return True
    header_units = _header_units(table)
    if not header_units:
        # Strong matrix signal (≥2 axis units): unit-less tables are incompatible.
        if len(axis_units) >= 2:
            return False
        # Single axis — keep soft (uncertain metadata).
        return True
    # Require at least one query axis unit to appear in table headers.
    return bool(axis_units & header_units)


def _attribute_hard_match(
    profile: TableApplicability,
    constraints: QueryConstraints,
) -> Optional[bool]:
    """True/False when query explicitly states an attribute the table declares."""
    decisions: List[bool] = []
    for c in constraints.constraints:
        if c.evidence_type != "explicit" or c.role in {ROLE_AXIS, "compare", "frequency"}:
            continue
        if c.key in {"axis_value", "compare_amount", "frequency", "numeric"}:
            continue
        key = c.key.lower()
        q_val = (c.value or "").strip().lower()
        if not q_val:
            continue
        # Match against declared attributes.
        if key in profile.attributes:
            attr_val = profile.attributes[key].lower()
            decisions.append(q_val in attr_val or attr_val in q_val or bool(_tokens(q_val) & _tokens(attr_val)))
            continue
        # Match against cue tokens for classification / period style values.
        q_toks = _tokens(q_val) | _tokens(c.evidence)
        if q_toks and profile.cue_tokens:
            if q_toks & profile.cue_tokens:
                decisions.append(True)
            elif key in {"reporting_period", "classification", "material_grade"}:
                # Explicit period/class with no overlap → incompatible.
                decisions.append(False)
    if not decisions:
        return None
    return all(decisions)


def select_eligible_table_chunks(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    get_table: Optional[Any] = None,
) -> SelectionDecision:
    """Filter chunks by structure + applicability; flag missing dims."""

    def _table_of(chunk: Dict[str, Any]) -> Dict[str, Any]:
        if get_table is not None:
            return get_table(chunk) or {}
        from app.rag.table_rag.ids import parse_table_payload

        raw = chunk.get("table")
        if isinstance(raw, dict) and (raw.get("rows") is not None or raw.get("headers")):
            return dict(raw)
        payload = parse_table_payload(chunk.get("table_json") or raw)
        return payload.to_dict() if payload else {}

    query = extract_query_constraints(question)
    profiles: List[Tuple[Dict[str, Any], Dict[str, Any], TableApplicability]] = []
    for chunk in chunks:
        table = _table_of(chunk)
        if not table and not chunk.get("table_id"):
            continue
        profiles.append((chunk, table, build_table_applicability(table, chunk=chunk)))

    if not profiles:
        return SelectionDecision(
            eligible_chunks=list(chunks),
            reason="no_profiles",
            query_constraints=query.to_dict(),
        )

    # Explicit Table N citation is resolved *after* the hard filters: a
    # question may cite a table in order to challenge it ("is adult Table 1
    # appropriate for this 8-year-old?"), and narrowing first would empty the
    # pool and hide the table that actually applies.
    table_num = question_table_number(question)

    duration_bounds = [
        float(c.numeric_value)
        for c in query.constraints
        if c.role == ROLE_EXCLUDE
        and c.numeric_value is not None
        and (
            (c.unit or "").lower() in _DURATION_UNIT_HINTS
            or re.search(r"(?i)year|yr|yo", c.evidence or query.raw_question)
        )
    ]
    # Always merge phrase-based cohort bounds (e.g. ">=12 years").
    for b in question_age_constraints(question):
        if b not in duration_bounds:
            duration_bounds.append(b)

    any_bound_metadata = any(
        (p.bound_ranges or p.age_ranges) for _, _, p in profiles
    )
    q_tokens = query.tokens or _tokens(question)

    eligible: List[Tuple[Dict[str, Any], Dict[str, Any], TableApplicability]] = []
    excluded: List[Dict[str, Any]] = []
    structure_blocked: List[Dict[str, Any]] = []

    for chunk, table, profile in profiles:
        validation = validate_table_structure(table)
        if validation.status == STRUCTURE_INVALID or not validation.reliable_for_lookup:
            structure_blocked.append(
                {
                    "table_id": profile.table_id,
                    "reason": "unreliable_table_structure",
                    "status": validation.status,
                    "reason_codes": list(validation.reason_codes),
                }
            )
            excluded.append(
                {
                    "table_id": profile.table_id,
                    "reason": "unreliable_table_structure",
                    "status": validation.status,
                    "reason_codes": list(validation.reason_codes),
                }
            )
            continue

        if not _axis_compatible(table, query):
            excluded.append(
                {
                    "table_id": profile.table_id,
                    "reason": "incompatible_header_units",
                    "query_units": sorted(_query_units(query)),
                    "table_units": sorted(profile.header_units),
                }
            )
            continue

        attr_match = _attribute_hard_match(profile, query)
        if attr_match is False:
            excluded.append(
                {
                    "table_id": profile.table_id,
                    "reason": "attribute_constraint_mismatch",
                }
            )
            continue

        # Cohort lexicon opposition (domain-agnostic): "adult" questions must
        # not keep pediatric-only siblings that then force applicability_bound
        # ambiguity against the adult matrix.
        q_cohort = q_tokens & {
            "adult",
            "adults",
            "adolescent",
            "adolescents",
            "pediatric",
            "paediatric",
            "child",
            "children",
            "infant",
            "infants",
        }
        t_cohort = set(profile.cue_tokens or set()) | set(profile.purpose_tokens or set())
        adultish = {"adult", "adults", "adolescent", "adolescents"}
        pedish = {
            "pediatric",
            "paediatric",
            "child",
            "children",
            "infant",
            "infants",
        }
        # A cited table's own population label is not a constraint on the
        # patient ("is adult Table 1 appropriate for this 8-year-old?").
        if (
            not table_num
            and (q_cohort & adultish)
            and (t_cohort & pedish)
            and not (t_cohort & adultish)
        ):
            excluded.append(
                {
                    "table_id": profile.table_id,
                    "reason": "cohort_lexicon_mismatch",
                }
            )
            continue
        if (
            not table_num
            and (q_cohort & pedish)
            and (t_cohort & adultish)
            and not (t_cohort & pedish)
        ):
            excluded.append(
                {
                    "table_id": profile.table_id,
                    "reason": "cohort_lexicon_mismatch",
                }
            )
            continue

        # Hard applicability-bound filter when query states a duration bound.
        ranges = profile.bound_ranges or profile.age_ranges
        if duration_bounds:
            if ranges:
                if not any(
                    _bound_compatible(bound, ranges) for bound in duration_bounds
                ):
                    excluded.append(
                        {
                            "table_id": profile.table_id,
                            "reason": "applicability_bound_mismatch",
                            "bounds": duration_bounds,
                        }
                    )
                    continue
            elif any_bound_metadata:
                excluded.append(
                    {
                        "table_id": profile.table_id,
                        "reason": "missing_bound_applicability_while_siblings_declare_it",
                        "bounds": duration_bounds,
                    }
                )
                continue

        eligible.append((chunk, table, profile))

    if table_num:
        cited = [
            (chunk, table, profile)
            for chunk, table, profile in eligible
            if table_number_match(table, chunk, table_num)
        ]
        if cited:
            eligible = cited
        elif eligible:
            # The cited table was filtered as inapplicable. Replace it with the
            # surviving table whose caption/purpose is closest to the question,
            # so the fallback is topical rather than arbitrary.
            probe = _condition_tokens(question) or (set(q_tokens) - _COHORT_TOKENS)
            eligible = _narrow_by_purpose_overlap(eligible, probe)

    mismatch_terms = _indication_mismatch_terms(question, eligible)
    if mismatch_terms:
        # The patient's condition is outside the scope of every candidate
        # table: this is a wrong-table situation, not missing metadata.
        return SelectionDecision(
            eligible_chunks=[],
            candidates=_eligibility_records(profiles, eligible, excluded),
            excluded=excluded,
            structure_blocked=structure_blocked,
            reason=(
                "indication mismatch — the stated patient condition ("
                + ", ".join(mismatch_terms)
                + ") is outside the scope of the candidate tables; the table "
                "used is not appropriate for this patient"
            ),
            requires_disambiguation=True,
            missing_dimensions=["indication"],
            query_constraints=query.to_dict(),
            mismatch_terms=mismatch_terms,
        )

    if not eligible:
        return SelectionDecision(
            eligible_chunks=[],
            candidates=_eligibility_records(profiles, [], excluded),
            excluded=excluded,
            structure_blocked=structure_blocked,
            reason="all_tables_excluded_by_hard_constraints",
            requires_disambiguation=True,
            missing_dimensions=["applicability"],
            query_constraints=query.to_dict(),
        )

    # Unique cue tokens per table relative to siblings.
    if len(eligible) >= 2:
        def _distinctive_tokens(profile: TableApplicability) -> Set[str]:
            # Attribute values + purpose/caption tokens are highest-signal for
            # class selection; do not rely only on the first few cue strings
            # (section labels are often appended after generic phrases).
            parts = list(profile.attributes.values()) + list(
                profile.differentiating_dimensions
            )
            parts.extend(
                r.raw for r in (profile.bound_ranges or profile.age_ranges) if r.raw
            )
            parts.extend(profile.cues[:12])
            toks = _tokens(" ".join(parts))
            toks |= attribute_value_tokens(profile)
            toks |= set(profile.purpose_tokens or set())
            return toks

        distinctive = [
            (chunk, table, profile, _distinctive_tokens(profile))
            for chunk, table, profile in eligible
        ]
        unique_hits: List[Tuple[Dict[str, Any], Dict[str, Any], TableApplicability, Set[str]]] = []
        for chunk, table, profile, tokens in distinctive:
            others: Set[str] = set()
            for _, _, _, other_tokens in distinctive:
                if other_tokens is tokens:
                    continue
                others |= other_tokens
            unique = tokens - others
            matched_unique = unique & q_tokens
            unique_hits.append((chunk, table, profile, matched_unique))

        if any(hit for _, _, _, hit in unique_hits):
            narrowed = [
                (chunk, table, profile)
                for chunk, table, profile, hit in unique_hits
                if hit
            ]
            if narrowed:
                eligible = narrowed

    # Detect sibling distinctness when query lacks differentiating cues.
    missing: List[str] = []
    requires = False
    if len(eligible) >= 2 and not duration_bounds:
        conflict_dims: List[str] = []
        distinct_pairs = 0
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                if profiles_are_distinct(eligible[i][2], eligible[j][2]):
                    distinct_pairs += 1
                    conflict_dims.extend(
                        conflicting_dimensions(eligible[i][2], eligible[j][2])
                    )
        if distinct_pairs:
            missing.extend(conflict_dims)
            for _, _, profile in eligible:
                missing.extend(profile.differentiating_dimensions)
            if not missing:
                missing = ["applicability_class"]
            q_unique_match = False
            for _, _, profile in eligible:
                others: Set[str] = set()
                for _, _, other in eligible:
                    if other is profile:
                        continue
                    others |= other.cue_tokens
                if (profile.cue_tokens - others) & q_tokens:
                    q_unique_match = True
                    break
            if not q_unique_match:
                requires = True

    # Also: even with duration bounds, conflicting non-bound dims need cues.
    # Coverage is value-aware: query tokens that hit attribute VALUES count
    # even when the dimension *name* never appears in the question.
    if len(eligible) >= 2:
        extra_dims: List[str] = []
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                dims = conflicting_dimensions(eligible[i][2], eligible[j][2])
                # Ignore pure bound dims when query already supplied bounds.
                if duration_bounds:
                    dims = [d for d in dims if d not in {"applicability_bound", "age_band"}]
                extra_dims.extend(dims)
        extra_dims = list(dict.fromkeys(extra_dims))
        if extra_dims:
            def _dim_name_or_key_hit(profile: TableApplicability, dim: str) -> bool:
                dim_toks = _tokens(dim.replace("_", " "))
                if dim_toks & q_tokens:
                    return True
                val = profile.attributes.get(dim.lower(), "") or profile.attributes.get(
                    dim, ""
                )
                if (_tokens(val) | dim_toks) & q_tokens:
                    return True
                return False

            # Profiles whose attribute VALUES / purpose tokens overlap the query.
            value_hits = [
                (chunk, table, profile)
                for chunk, table, profile in eligible
                if profile_query_attr_hits(profile, q_tokens)
                or bool(set(profile.purpose_tokens or set()) & q_tokens)
                or any(_dim_name_or_key_hit(profile, d) for d in extra_dims)
            ]
            explicit_dim_stated = any(
                c.key.lower() == d.lower()
                for d in extra_dims
                for c in query.constraints
                if c.evidence_type == "explicit"
            )
            covered = bool(value_hits) or explicit_dim_stated

            if covered and value_hits:
                # Narrow to tables whose attribute values the query actually states.
                # Prefer strict attribute-value hits over dim-name-only hits.
                strict = [
                    item
                    for item in value_hits
                    if profile_query_attr_hits(item[2], q_tokens)
                ]
                narrowed = strict or value_hits
                if narrowed:
                    eligible = narrowed

            if len(eligible) == 1:
                # Differentiating value selected a unique table — dims are covered.
                requires = False
                missing = [
                    d
                    for d in missing
                    if d not in extra_dims
                    and d not in {"applicability_class"}
                ]
            elif not covered:
                missing.extend(extra_dims)
                requires = True
            elif len(eligible) > 1:
                # Still multi-eligible after value narrowing — remain ambiguous.
                missing.extend(extra_dims)
                requires = True

    # Provenance-only missing dims never force hard disambiguation.
    missing = applicability_dimensions(list(dict.fromkeys(missing)))
    if requires and not missing and len(eligible) <= 1:
        requires = False
    # A single remaining eligible table is uniquely resolved.
    if len(eligible) <= 1:
        requires = False
        missing = []

    # Build per-dim attribute values for value-aware missing-dimension marking.
    attr_values_by_dim: Dict[str, List[str]] = {}
    for _, _, profile in eligible:
        for key, val in (profile.attributes or {}).items():
            if not key or not val:
                continue
            attr_values_by_dim.setdefault(str(key).lower(), []).append(str(val))

    query = mark_missing_dimensions(
        query,
        missing,
        attribute_values_by_dim=attr_values_by_dim,
    )

    return SelectionDecision(
        eligible_chunks=[c for c, _, _ in eligible],
        candidates=_eligibility_records(profiles, eligible, excluded),
        excluded=excluded,
        structure_blocked=structure_blocked,
        reason="ok" if not requires else "ambiguous_applicability",
        requires_disambiguation=requires,
        missing_dimensions=list(dict.fromkeys(query.missing_dimensions or missing)),
        query_constraints=query.to_dict(),
    )
