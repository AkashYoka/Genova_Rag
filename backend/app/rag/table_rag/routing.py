"""Query routing: paragraph vs table vs both modalities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryModality(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    BOTH = "both"


# Generic structural cues only — no product/indication/unit vocab.
_TABLE_CUE_RE = re.compile(
    r"(?i)\b(table|chart|grid|matrix|schedule|lookup|row|column|cell|intersection)\b"
)
_NUMERIC_RE = re.compile(r"(?i)\b\d+(?:\.\d+)?\b")
_PARAGRAPH_CUE_RE = re.compile(
    r"(?i)\b("
    r"what\s+is|explain|describe|summarize|summary|"
    r"how\s+(?:does|do|is|are)|why|"
    r"definition|overview|background|introduction"
    r")\b"
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
    """Classify whether a question needs paragraph, table, or both retrieval."""
    q = (question or "").strip()
    if not q:
        return QueryRoute(QueryModality.PARAGRAPH, "empty_query")

    table_cue = bool(_TABLE_CUE_RE.search(q))
    numeric_count = len(_NUMERIC_RE.findall(q))
    paragraph_cue = bool(_PARAGRAPH_CUE_RE.search(q))
    multi_numeric = numeric_count >= 2
    structured = table_cue and multi_numeric

    if structured:
        if paragraph_cue:
            return QueryRoute(
                QueryModality.BOTH,
                "table_numeric_plus_prose",
                wants_structured_lookup=True,
            )
        return QueryRoute(
            QueryModality.TABLE,
            "table_numeric_lookup",
            wants_structured_lookup=True,
        )

    if table_cue and numeric_count >= 1:
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
