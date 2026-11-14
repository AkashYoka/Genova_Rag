"""Query routing: paragraph vs table vs both modalities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QueryModality(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    BOTH = "both"


_TABLE_CUE_RE = re.compile(
    r"(?i)\b("
    r"table|chart|grid|matrix|schedule|lookup|"
    r"row|column|cell|intersection|"
    r"frequency|every\s+\d+\s+(?:weeks?|days?|hours?|months?)|"
    r"amount|rate|threshold|parameter|range|"
    r"compare|versus|vs\.?|higher|lower|maximum|minimum|total"
    r")\b"
)
_NUMERIC_UNIT_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*(?:[a-zA-Zμµ%/²³°$]{1,12}|"
    r"weeks?|days?|hours?|months?)\b"
)
_PARAGRAPH_CUE_RE = re.compile(
    r"(?i)\b("
    r"what\s+is|explain|describe|summarize|summary|mechanism|"
    r"how\s+(?:does|do|is|are)|why|"
    r"definition|overview|background|introduction"
    r")\b"
)
# Two distinct numeric+unit mentions → likely matrix axis intersection.
_UNIT_TOKEN_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*([a-zA-Zμµ%/²³]{1,8})\b"
)


@dataclass(frozen=True)
class QueryRoute:
    modality: QueryModality
    reason: str
    wants_structured_lookup: bool = False

    @property
    def needs_tables(self) -> bool:
        return self.modality in (QueryModality.TABLE, QueryModality.BOTH)

    @property
    def needs_paragraphs(self) -> bool:
        return self.modality in (QueryModality.PARAGRAPH, QueryModality.BOTH)


def route_query(question: str) -> QueryRoute:
    """Classify whether a question needs paragraph, table, or both retrieval.

    Domain-agnostic heuristics reuse the same numeric/table cues already used
    by structured lookup, without hardcoding product names.
    """
    q = (question or "").strip()
    if not q:
        return QueryRoute(QueryModality.PARAGRAPH, "empty_query")

    table_cue = bool(_TABLE_CUE_RE.search(q))
    numeric = _NUMERIC_UNIT_RE.findall(q)
    paragraph_cue = bool(_PARAGRAPH_CUE_RE.search(q))
    unit_tokens = {
        m.group(1).lower().rstrip("s")
        for m in _UNIT_TOKEN_RE.finditer(q)
        if m.group(1)
    }
    axis_pair = len(unit_tokens) >= 2
    multi_numeric = len(numeric) >= 2

    structured = axis_pair or (table_cue and multi_numeric)

    if structured or (table_cue and multi_numeric):
        if paragraph_cue:
            return QueryRoute(
                QueryModality.BOTH,
                "table_numeric_plus_prose",
                wants_structured_lookup=True,
            )
        return QueryRoute(
            QueryModality.TABLE,
            "numeric_or_matrix_lookup",
            wants_structured_lookup=True,
        )

    if table_cue and numeric:
        return QueryRoute(
            QueryModality.TABLE,
            "table_cue_with_value",
            wants_structured_lookup=True,
        )

    if table_cue and paragraph_cue:
        return QueryRoute(QueryModality.BOTH, "table_and_prose_cues")

    if table_cue:
        return QueryRoute(QueryModality.TABLE, "explicit_table_language")

    if multi_numeric:
        # Numerical question without explicit "table" — still prefer tables.
        if paragraph_cue:
            return QueryRoute(
                QueryModality.BOTH,
                "numeric_plus_prose",
                wants_structured_lookup=True,
            )
        return QueryRoute(
            QueryModality.TABLE,
            "numerical_question",
            wants_structured_lookup=True,
        )

    if paragraph_cue:
        return QueryRoute(QueryModality.PARAGRAPH, "prose_question")

    return QueryRoute(QueryModality.BOTH, "default_both")


def route_query_modality(question: str) -> QueryModality:
    """Convenience wrapper returning only the modality enum."""
    return route_query(question).modality
