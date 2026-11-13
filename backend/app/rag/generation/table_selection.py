"""Domain-agnostic table eligibility and multi-table conflict detection.

Compares query constraints to table applicability metadata without hardcoded
domain schemas. Sibling tables that share numeric axes but differ in
applicability must not be silently collapsed to one answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
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


def question_age_constraints(question: str) -> List[float]:
    """Back-compat: calendar-duration ROLE_EXCLUDE numerics from the question."""
    bounds: List[float] = []
    q = question or ""
    for c in extract_numeric_constraints(q):
        if c.role != ROLE_EXCLUDE:
            continue
        pattern = (
            rf"(?i){re.escape(str(int(c.value) if float(c.value).is_integer() else c.value))}"
            rf"\s*-?\s*(?:year|yr|yo)"
        )
        if re.search(pattern, q) or c.unit.lower() in _DURATION_UNIT_HINTS:
            bounds.append(float(c.value))
    return bounds


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


def _range_midpoint(parsed: ParsedRange) -> Optional[float]:
    if parsed.low is not None and parsed.high is not None:
        return (parsed.low + parsed.high) / 2.0
    if parsed.low is not None:
        return parsed.low + 1.0
    if parsed.high is not None:
        return max(0.0, parsed.high - 1.0)
    return None


def profiles_are_distinct(
    left: TableApplicability,
    right: TableApplicability,
    *,
    max_overlap: float = 0.35,
) -> bool:
    """True when two tables look like different applicability classes."""
    left_ranges = left.bound_ranges or left.age_ranges
    right_ranges = right.bound_ranges or right.age_ranges
    if left_ranges and right_ranges:
        shared = False
        for a in left_ranges:
            mid = _range_midpoint(a)
            if mid is None:
                continue
            if any(
                range_contains(b, mid, unit=b.unit or a.unit or "years")
                for b in right_ranges
            ):
                shared = True
                break
        if not shared:
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
    shared_keys = set(left.attributes) & set(right.attributes)
    for key in shared_keys:
        if left.attributes[key].lower() != right.attributes[key].lower():
            return True
    # Asymmetric attributes with non-trivial unique tokens.
    only_left = set(left.attributes) - set(right.attributes)
    only_right = set(right.attributes) - set(left.attributes)
    if (only_left or only_right) and (left.attributes or right.attributes):
        if left.cue_tokens and right.cue_tokens and overlap <= 0.6:
            return True
    return False


def conflicting_dimensions(
    left: TableApplicability,
    right: TableApplicability,
) -> List[str]:
    """Return attribute keys / dims that distinguish two profiles."""
    dims: List[str] = []
    shared_keys = set(left.attributes) & set(right.attributes)
    for key in shared_keys:
        if left.attributes[key].lower() != right.attributes[key].lower():
            dims.append(key)
    left_only = set(left.attributes) - set(right.attributes)
    right_only = set(right.attributes) - set(left.attributes)
    # Unique attribute keys are differentiating when both sides have attrs.
    if left.attributes and right.attributes:
        dims.extend(sorted(left_only | right_only))
    left_ranges = left.bound_ranges or left.age_ranges
    right_ranges = right.bound_ranges or right.age_ranges
    if left_ranges and right_ranges and profiles_are_distinct(left, right):
        dims.append("applicability_bound")
    for d in left.differentiating_dimensions + right.differentiating_dimensions:
        if d and d not in dims:
            dims.append(d)
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
    """Reject tables whose headers cannot host the query's unit-bearing axes."""
    axis_units = {
        (c.unit or "").lower().replace(" ", "")
        for c in constraints.constraints
        if c.role == ROLE_AXIS and c.unit
    }
    if len(axis_units) < 1:
        return True
    header_units = _header_units(table)
    if not header_units:
        # No unit evidence in headers — do not hard-reject.
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
        raw = chunk.get("table")
        if isinstance(raw, dict):
            return raw
        import json

        tj = chunk.get("table_json")
        if isinstance(tj, dict):
            return tj
        if isinstance(tj, str) and tj.strip():
            try:
                data = json.loads(tj)
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

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
    # Fallback to legacy helper for tests / phrasing edge cases.
    if not duration_bounds:
        duration_bounds = question_age_constraints(question)

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

    if not eligible:
        return SelectionDecision(
            eligible_chunks=[],
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
        excluded=excluded,
        structure_blocked=structure_blocked,
        reason="ok" if not requires else "ambiguous_applicability",
        requires_disambiguation=requires,
        missing_dimensions=list(dict.fromkeys(query.missing_dimensions or missing)),
        query_constraints=query.to_dict(),
    )
