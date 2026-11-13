"""Domain-agnostic query constraint extraction for structured table lookup.

Extracts explicit/inferred/missing attributes without hardcoded domain fields.

Intentionally deterministic (regex / numeric roles) — not an LLM prompt — to
avoid a redundant model call on every query. Soft not-found / verified-cell
prompt blocks in ``structured_lookup`` guide answer generation when axes miss.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from app.rag.generation.structured_lookup import (
    ROLE_AXIS,
    ROLE_COMPARE,
    ROLE_EXCLUDE,
    ROLE_FREQUENCY,
    NumericConstraint,
    extract_numeric_constraints,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[/-][a-z0-9]+)?", re.I)
_PERIOD_RE = re.compile(
    r"(?i)\b(FY\s?\d{2,4}|Q[1-4]\s?\d{2,4}|calendar\s+year\s+\d{4}|CY\s?\d{4})\b"
)
_GRADE_RE = re.compile(
    r"(?i)\b((?:ASTM|ISO|EN|DIN)\s*[A-Z0-9][-A-Z0-9.]*)\b"
)
_KEY_VALUE_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9_\-]{2,24})\s*(?:=|:)\s*([a-z0-9][a-z0-9_\-./]{0,40})"
)

# Evidence types
EVIDENCE_EXPLICIT = "explicit"
EVIDENCE_INFERRED = "inferred"
EVIDENCE_MISSING = "missing"

_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "what",
        "which",
        "table",
        "value",
        "values",
        "according",
        "recommended",
        "patient",
        "patients",
    }
)


@dataclass
class ExtractedConstraint:
    """One query constraint with provenance (domain-agnostic key/value)."""

    key: str
    value: str
    kind: str = "categorical"  # categorical | numeric | range | unit | period
    evidence_type: str = EVIDENCE_EXPLICIT
    confidence: float = 0.8
    evidence: str = ""
    role: str = "applicability"  # axis | exclude | compare | frequency | applicability
    unit: str = ""
    numeric_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryConstraints:
    constraints: List[ExtractedConstraint] = field(default_factory=list)
    numeric: List[NumericConstraint] = field(default_factory=list)
    explicit_keys: Set[str] = field(default_factory=set)
    missing_dimensions: List[str] = field(default_factory=list)
    tokens: Set[str] = field(default_factory=set)
    raw_question: str = ""

    def values_for(self, key: str) -> List[str]:
        key_l = (key or "").lower()
        return [
            c.value
            for c in self.constraints
            if c.key.lower() == key_l and c.evidence_type != EVIDENCE_MISSING
        ]

    def has_explicit(self, key: str) -> bool:
        return (key or "").lower() in self.explicit_keys

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": [c.to_dict() for c in self.constraints],
            "explicit_keys": sorted(self.explicit_keys),
            "missing_dimensions": list(self.missing_dimensions),
            "tokens": sorted(self.tokens),
            "raw_question": self.raw_question,
            "numeric": [
                {"value": n.value, "unit": n.unit, "role": n.role}
                for n in self.numeric
            ],
        }


def _tokens(text: str) -> Set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) > 2 and t.lower() not in _STOP
    }


def extract_query_constraints(question: str) -> QueryConstraints:
    """Parse structured constraints from a natural-language question."""
    q = question or ""
    constraints: List[ExtractedConstraint] = []
    numeric = extract_numeric_constraints(q)

    for n in numeric:
        if n.role == ROLE_AXIS:
            key = "axis_value"
            kind = "numeric"
            role = ROLE_AXIS
        elif n.role == ROLE_EXCLUDE:
            # Calendar-duration excludes are applicability bounds, not axes.
            key = "applicability_bound"
            kind = "numeric"
            role = ROLE_EXCLUDE
        elif n.role == ROLE_FREQUENCY:
            key = "frequency"
            kind = "numeric"
            role = ROLE_FREQUENCY
        elif n.role == ROLE_COMPARE:
            key = "compare_amount"
            kind = "numeric"
            role = ROLE_COMPARE
        else:
            key = "numeric"
            kind = "numeric"
            role = n.role
        constraints.append(
            ExtractedConstraint(
                key=key,
                value=f"{n.value}{(' ' + n.unit) if n.unit else ''}".strip(),
                kind=kind,
                evidence_type=EVIDENCE_EXPLICIT,
                confidence=0.9,
                evidence=q,
                role=role,
                unit=n.unit,
                numeric_value=float(n.value),
            )
        )

    for m in _PERIOD_RE.finditer(q):
        constraints.append(
            ExtractedConstraint(
                key="reporting_period",
                value=m.group(1).strip(),
                kind="period",
                evidence_type=EVIDENCE_EXPLICIT,
                confidence=0.95,
                evidence=m.group(0),
                role="applicability",
            )
        )

    for m in _GRADE_RE.finditer(q):
        constraints.append(
            ExtractedConstraint(
                key="classification",
                value=m.group(1).strip(),
                kind="categorical",
                evidence_type=EVIDENCE_EXPLICIT,
                confidence=0.9,
                evidence=m.group(0),
                role="applicability",
            )
        )

    for m in _KEY_VALUE_RE.finditer(q):
        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        if key in _STOP:
            continue
        constraints.append(
            ExtractedConstraint(
                key=key,
                value=value,
                kind="categorical",
                evidence_type=EVIDENCE_EXPLICIT,
                confidence=0.75,
                evidence=m.group(0),
                role="applicability",
            )
        )

    explicit_keys = {
        c.key.lower()
        for c in constraints
        if c.evidence_type == EVIDENCE_EXPLICIT
    }
    return QueryConstraints(
        constraints=constraints,
        numeric=numeric,
        explicit_keys=explicit_keys,
        tokens=_tokens(q),
        raw_question=q,
    )


def mark_missing_dimensions(
    query: QueryConstraints,
    candidate_dimensions: Sequence[str],
    *,
    attribute_values_by_dim: Optional[Dict[str, Sequence[str]]] = None,
) -> QueryConstraints:
    """Attach dimensions that candidates require but the query does not state.

    A dimension is not missing when the query already contains tokens from that
    dimension's **attribute values** across candidates (e.g. query says
    ``retail`` while the dim key is ``reporting_segment``), not only when the
    dimension *name* appears in the question.
    """
    values_by_dim = {
        str(k).strip().lower(): list(v or [])
        for k, v in (attribute_values_by_dim or {}).items()
        if str(k).strip()
    }
    # Flat token set of all candidate attribute values (fallback when dim key
    # from differentiating_dimensions does not match an attribute key).
    all_value_tokens: Set[str] = set()
    for vals in values_by_dim.values():
        for val in vals:
            all_value_tokens |= _tokens(str(val))

    missing: List[str] = []
    for dim in candidate_dimensions:
        dim_l = str(dim or "").strip().lower()
        if not dim_l:
            continue
        if dim_l in query.explicit_keys:
            continue
        # Token overlap with dimension name or already-extracted values.
        dim_tokens = _tokens(dim_l.replace("_", " "))
        if dim_tokens & query.tokens:
            continue
        # Value-aware: query mentions an attribute value for this dim key.
        dim_vals = values_by_dim.get(dim_l) or []
        dim_val_tokens: Set[str] = set()
        for val in dim_vals:
            dim_val_tokens |= _tokens(str(val))
        if dim_val_tokens & query.tokens:
            continue
        # Differentiating dim name may not match attribute keys — if the query
        # hits any candidate attribute value, treat the dim as stated.
        if not dim_vals and all_value_tokens & query.tokens:
            continue
        # Special-case: applicability_bound covers calendar-year population dims.
        if dim_l in {"age_band", "population"} and any(
            c.key == "applicability_bound" for c in query.constraints
        ):
            continue
        if dim_l == "reporting_period" and any(
            c.key == "reporting_period" for c in query.constraints
        ):
            continue
        missing.append(str(dim).strip())
        query.constraints.append(
            ExtractedConstraint(
                key=str(dim).strip(),
                value="",
                kind="categorical",
                evidence_type=EVIDENCE_MISSING,
                confidence=0.0,
                evidence="",
                role="applicability",
            )
        )
    query.missing_dimensions = list(dict.fromkeys(missing))
    return query
