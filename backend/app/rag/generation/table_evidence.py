"""Domain-agnostic table evidence contracts for answer-time RAG.

Holds verified / insufficient evidence objects and the fail-closed answer-state
machine used by the agent. No regex cell lookup and no domain lexicons —
cell selection happens in ``llm_table_reasoning``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

VERIFIED_BLOCK_MARKER = "CRITICAL — Verified table evidence is authoritative."

STATUS_MATCHED = "matched"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_NOT_FOUND = "not_found"
STATUS_OUT_OF_RANGE = "out_of_range"
STATUS_INVALID_TABLE_STRUCTURE = "invalid_table_structure"
STATUS_SELECTOR_FAILED = "selector_failed"
STATUS_VERIFICATION_FAILED = "verification_failed"
STATUS_EMPTY_CELL = "empty_cell"

ANSWER_STATE_VERIFIED = "verified"
ANSWER_STATE_AMBIGUOUS = "ambiguous"
ANSWER_STATE_INSUFFICIENT = "insufficient"
ANSWER_STATE_INVALID = "invalid"
ANSWER_STATE_NOT_APPLICABLE = "not_applicable"

_INVALID_LOOKUP_STATUSES = frozenset(
    {
        STATUS_INVALID_TABLE_STRUCTURE,
        STATUS_VERIFICATION_FAILED,
        STATUS_EMPTY_CELL,
        STATUS_SELECTOR_FAILED,
        STATUS_OUT_OF_RANGE,
    }
)
_AMBIGUOUS_LOOKUP_STATUSES = frozenset(
    {STATUS_AMBIGUOUS, "needs_disambiguation", "ambiguous_applicability"}
)
_VERIFIED_LOOKUP_STATUSES = frozenset(
    {"", STATUS_MATCHED, "calculated", "verified", "answered"}
)


@dataclass
class AnswerStateDecision:
    """One authoritative answer state plus the evidence that controls it."""

    state: str = ANSWER_STATE_NOT_APPLICABLE
    status: str = ""
    soft: bool = False
    source: Optional[Dict[str, Any]] = None
    verified: Optional[Dict[str, Any]] = None

    @property
    def blocks_definitive_answer(self) -> bool:
        return self.state in {ANSWER_STATE_AMBIGUOUS, ANSWER_STATE_INVALID}


def answer_state_for_status(status: str, *, soft: bool = False) -> str:
    """Map one evidence status to the authoritative answer state."""
    text = str(status or "").strip().lower()
    if text in _AMBIGUOUS_LOOKUP_STATUSES:
        return ANSWER_STATE_AMBIGUOUS
    if text in _INVALID_LOOKUP_STATUSES:
        return ANSWER_STATE_INVALID
    if text in _VERIFIED_LOOKUP_STATUSES and not soft:
        return ANSWER_STATE_VERIFIED
    return ANSWER_STATE_INSUFFICIENT


def answer_state_from_sources(
    sources: Sequence[Dict[str, Any]] | None,
) -> AnswerStateDecision:
    """Derive the single authoritative answer state from evidence sources.

    Fail closed: a hard table-evidence failure outranks any verified cell.
    Ordinary prose retrieval (no table evidence) is ``not_applicable``.
    """
    items = [s for s in (sources or []) if isinstance(s, dict)]
    hard = [
        s
        for s in items
        if s.get("insufficient_table_evidence") and not s.get("soft_insufficient")
    ]
    if hard:
        status = str(hard[0].get("status") or hard[0].get("verification_status") or "")
        return AnswerStateDecision(
            state=answer_state_for_status(status),
            status=status,
            soft=False,
            source=hard[0],
        )
    for src in items:
        meta = src.get("verified_cell") or src.get("verified_calculation")
        if not isinstance(meta, dict) or meta.get("value") is None:
            continue
        status = str(meta.get("status") or STATUS_MATCHED).strip().lower()
        if status in _VERIFIED_LOOKUP_STATUSES:
            return AnswerStateDecision(
                state=ANSWER_STATE_VERIFIED,
                status=status or STATUS_MATCHED,
                verified=meta,
            )
        return AnswerStateDecision(
            state=answer_state_for_status(status),
            status=status,
            source=meta,
        )
    soft_src = next(
        (
            s
            for s in items
            if s.get("insufficient_table_evidence") and s.get("soft_insufficient")
        ),
        None,
    )
    if soft_src is not None:
        return AnswerStateDecision(
            state=ANSWER_STATE_INSUFFICIENT,
            status=str(soft_src.get("status") or ""),
            soft=True,
            source=soft_src,
        )
    return AnswerStateDecision(state=ANSWER_STATE_NOT_APPLICABLE)


@dataclass
class VerifiedCell:
    value: str
    row_headers: List[str]
    col_headers: List[str]
    row_index: int
    col_index: int
    page: Optional[int] = None
    filename: str = ""
    confidence: float = 0.0
    table_headers: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: str = STATUS_MATCHED
    table_id: str = ""
    normalized_value: str = ""
    document_version: str = ""
    structural_validation: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def provenance_record(self) -> Dict[str, Any]:
        evidence = self.evidence or {}
        return {
            "document_id": str(
                evidence.get("source_document_id")
                or evidence.get("document_id")
                or ""
            ),
            "document_version": str(
                self.document_version or evidence.get("document_version") or ""
            ),
            "page": self.page,
            "table_id": self.table_id,
            "chunk_id": self.chunk_id or str(evidence.get("chunk_id") or ""),
            "table_caption": str(evidence.get("table_caption") or ""),
            "row_index": self.row_index,
            "col_index": self.col_index,
            "original_cell_text": str(
                evidence.get("original_cell_text") or self.value
            ),
            "normalized_row_headers": list(self.row_headers or []),
            "normalized_column_headers": list(self.col_headers or []),
            "structural_validation": dict(
                self.structural_validation
                or evidence.get("structural_validation")
                or {}
            ),
            "verification_status": self.status,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "row_headers": list(self.row_headers or []),
            "col_headers": list(self.col_headers or []),
            "row_index": self.row_index,
            "col_index": self.col_index,
            "page": self.page,
            "filename": self.filename,
            "confidence": self.confidence,
            "table_headers": list(self.table_headers or []),
            "evidence": dict(self.evidence or {}),
            "status": self.status,
            "table_id": self.table_id,
            "normalized_value": self.normalized_value,
            "document_version": self.document_version,
            "structural_validation": dict(self.structural_validation or {}),
            "chunk_id": self.chunk_id,
            "provenance": self.provenance_record(),
        }

    def to_prompt_block(self) -> str:
        row_s = (
            " | ".join(self.row_headers)
            if self.row_headers
            else f"row {self.row_index}"
        )
        col_s = (
            " | ".join(self.col_headers)
            if self.col_headers
            else f"col {self.col_index}"
        )
        loc = []
        if self.filename:
            loc.append(self.filename)
        if self.page is not None:
            loc.append(f"Page {self.page}")
        where = ", ".join(loc) if loc else "retrieved table"
        caption = str(self.evidence.get("table_caption") or "").strip()
        section = str(self.evidence.get("table_section") or "").strip()
        tid = self.table_id or "unknown"
        provenance = (
            f"- Source: {where}\n"
            f"- Table id: {tid}\n"
            + (f"- Caption: {caption}\n" if caption else "")
            + (f"- Section: {section}\n" if section else "")
            + "- Provenance rule: cite THIS page and table only for the "
            "matrix value. Never attribute this cell to a different page/table.\n"
        )
        stated = self.evidence.get("prescribed_value")
        if stated is not None:
            stated_unit = self.evidence.get("prescribed_unit") or ""
            stated_s = f"{stated}{(' ' + stated_unit) if stated_unit else ''}".strip()
            return (
                f"{VERIFIED_BLOCK_MARKER}\n"
                "This question compares a stated amount against a verified table cell.\n"
                f"- Label / table cell value (authoritative): {self.value}\n"
                f"- Stated amount: {stated_s}\n"
                f"- Row: {row_s}\n"
                f"- Column: {col_s}\n"
                f"{provenance}"
                "Describe the comparison factually. Do not invent classifications "
                "beyond what the evidence supports. Do not replace the label cell "
                "with a neighboring cell or another table.\n"
                "Preserve value and units exactly as verified."
            )
        companions = self.evidence.get("same_row_companions") or []
        companion_lines = ""
        if companions:
            bits = [
                f"{c.get('header')}: {c.get('value')}"
                for c in companions
                if isinstance(c, dict) and c.get("value")
            ]
            if bits:
                companion_lines = (
                    "- Same-row companions (use these exact texts for "
                    "schedule/frequency/axis labels; do not invent from another row): "
                    + "; ".join(bits)
                    + "\n"
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
            f"{companion_lines}"
            f"{provenance}"
            f"- Status: {self.status}\n"
            "- Coordinate verification: python_sql_hydrate\n"
            "Use only this verified cell for the matrix value. If the question's "
            "applicability does not match this table's caption/scope, say so "
            "explicitly rather than picking a different unverified table."
        )


@dataclass
class InsufficientEvidence:
    reason: str
    structured_lookup: bool = True
    status: str = STATUS_NOT_FOUND
    soft: bool = False
    candidate_values: List[str] = field(default_factory=list)
    candidate_table_ids: List[str] = field(default_factory=list)
    missing_dimensions: List[str] = field(default_factory=list)
    structural_validation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "structured_lookup": self.structured_lookup,
            "status": self.status,
            "soft": self.soft,
            "candidate_values": list(self.candidate_values or []),
            "candidate_table_ids": list(self.candidate_table_ids or []),
            "missing_dimensions": list(self.missing_dimensions or []),
            "structural_validation": dict(self.structural_validation or {}),
        }

    def to_prompt_block(self) -> str:
        if self.status == STATUS_AMBIGUOUS:
            dims = (
                ", ".join(self.missing_dimensions)
                if self.missing_dimensions
                else "applicability / population / period / category"
            )
            return (
                "AMBIGUOUS table evidence — HARD ABSTAIN.\n"
                f"- Reason: {self.reason}\n"
                f"- Status: {self.status}\n"
                f"- Missing or conflicting applicability dimensions: {dims}\n"
                f"- Candidate values (NOT verified — do not treat as facts): "
                f"{self.candidate_values or []}\n"
                f"- Candidate tables: {self.candidate_table_ids or []}\n"
                "- Your entire answer must be inconclusive/clarification only.\n"
                "- Do NOT invent a definitive matrix conclusion.\n"
                "- Ask for the missing applicability constraint or list candidates "
                "with provenance without choosing one."
            )
        if self.soft:
            return (
                "SOFT table-lookup notice (not a hard abstain):\n"
                f"- Reason: {self.reason}\n"
                f"- Status: {self.status}\n"
                "- A unique matrix cell was not verified. Answer from clear prose "
                "guidance in Document Context when present.\n"
                "- Do not invent a matrix cell, swap sibling tables, or fabricate "
                "row/column intersections.\n"
                "- Preserve uncertainty: separate labeled facts from interpretation."
            )
        return (
            "Insufficient table evidence:\n"
            f"- Reason: {self.reason}\n"
            f"- Status: {self.status}\n"
            "- Do not invent a cell value from flattened markdown or neighboring "
            "rows/columns.\n"
            "- If you cannot verify a unique row-column intersection, say evidence "
            "is insufficient."
        )
