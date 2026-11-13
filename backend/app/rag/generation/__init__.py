"""Answer-time table interpretation and verified cell lookup."""

from app.rag.generation.structured_lookup import (
    STATUS_AMBIGUOUS,
    STATUS_MATCHED,
    STATUS_NOT_FOUND,
    STATUS_OUT_OF_RANGE,
    STATUS_INVALID_TABLE_STRUCTURE,
    InsufficientEvidence,
    VerifiedCell,
    lookup_cell_in_table,
)
from app.rag.generation.table_interpretation import interpret_table_question
from app.rag.generation.table_selection import select_eligible_table_chunks
from app.rag.generation.table_selector import (
    interpret_with_selector,
    verify_lookup_decision,
)

__all__ = [
    "STATUS_AMBIGUOUS",
    "STATUS_MATCHED",
    "STATUS_NOT_FOUND",
    "STATUS_OUT_OF_RANGE",
    "STATUS_INVALID_TABLE_STRUCTURE",
    "InsufficientEvidence",
    "VerifiedCell",
    "interpret_table_question",
    "interpret_with_selector",
    "lookup_cell_in_table",
    "select_eligible_table_chunks",
    "verify_lookup_decision",
]
