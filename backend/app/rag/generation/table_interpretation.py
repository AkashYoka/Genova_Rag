"""Orchestrate structured table lookup over retrieved RAG chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from app.rag.generation.structured_lookup import (
    InsufficientEvidence,
    VerifiedCell,
    lookup_from_chunk,
)

_STRUCTURED_LOOKUP_RE = re.compile(
    r"(?i)\b("
    r"table|chart|grid|matrix|intersection|lookup|"
    r"row|column|cell|"
    r"dosing\s+table|dose\s+table|dosage\s+table"
    r")\b"
)
# Strong dual-axis cue: weight-like unit + assay/concentration-like unit.
_AXIS_PAIR_RE = re.compile(
    r"(?is)(?=.*\b\d+(?:\.\d+)?\s*kg\b)(?=.*\b\d+(?:\.\d+)?\s*(?:iu/?mL|ng/?mL|ug/?mL|µg/?mL|μg/?mL))"
)
_NUMERIC_HINT_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*(?:mg|kg|g|ml|l|%|iu/?mL|ppm)\b"
)


@dataclass
class TableInterpretationResult:
    is_structured_lookup: bool
    result: Optional[Union[VerifiedCell, InsufficientEvidence]] = None
    prompt_prefix: str = ""
    sources_update: Dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> Optional[VerifiedCell]:
        return self.result if isinstance(self.result, VerifiedCell) else None

    @property
    def insufficient(self) -> Optional[InsufficientEvidence]:
        return self.result if isinstance(self.result, InsufficientEvidence) else None


def is_structured_lookup_question(question: str) -> bool:
    """True only for questions that likely need a matrix cell intersection.

    Domain-agnostic: requires explicit table language OR dual-axis numeric cues
    (e.g. weight kg + assay unit). Ordinary dose/route/AE classification prose
    questions should NOT force structured lookup / insufficient-table banners.
    """
    q = (question or "").strip()
    if not q:
        return False
    if _STRUCTURED_LOOKUP_RE.search(q):
        return True
    if _AXIS_PAIR_RE.search(q):
        return True
    # Fallback: ≥2 distinct unit-bearing numerics including a weight unit.
    nums = _NUMERIC_HINT_RE.findall(q)
    joined = " ".join(nums).lower()
    return len(nums) >= 2 and "kg" in joined and any(
        u in joined for u in ("iu", "ng/ml", "ug/ml", "µg/ml", "μg/ml", "%")
    )


def _table_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for chunk in chunks or []:
        chunk_type = str(
            chunk.get("chunk_type") or chunk.get("content_type") or ""
        ).lower()
        text = str(chunk.get("text") or "")
        if chunk.get("table_json") or chunk_type == "table" or "table facts:" in text.lower():
            out.append(chunk)
    return out


def interpret_table_question(
    question: str,
    chunks: List[Dict[str, Any]],
) -> TableInterpretationResult:
    """Run verified cell lookup when the question looks like a table query.

    Returns a prompt prefix for verified evidence or insufficient-evidence abstain.
    Non-lookup questions return ``is_structured_lookup=False`` with empty prefix.
    """
    if not is_structured_lookup_question(question):
        return TableInterpretationResult(is_structured_lookup=False)

    table_chunks = _table_chunks(chunks)
    if not table_chunks:
        # Soft fail: do not inject abstain boilerplate into the LLM prompt.
        # Let prose RAG answer; callers may still inspect sources_update.
        insuff = InsufficientEvidence("no table chunks retrieved for structured lookup")
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=insuff,
            prompt_prefix="",
            sources_update={
                "insufficient_table_evidence": True,
                "reason": insuff.reason,
                "soft_insufficient": True,
            },
        )

    def _chunk_overlap(chunk: Dict[str, Any]) -> float:
        raw_tokens = [
            t.lower()
            for t in re.findall(r"[a-zA-Z0-9]+", question or "")
            if len(t) > 1
        ]
        q_tokens = {t for t in raw_tokens if len(t) > 2}
        if not q_tokens:
            return 0.0
        blob = " ".join(
            str(x or "")
            for x in (
                chunk.get("text"),
                chunk.get("section"),
                chunk.get("subsection"),
                chunk.get("filename"),
            )
        ).lower()
        blob_tokens = [
            t.lower() for t in re.findall(r"[a-zA-Z0-9]+", blob) if len(t) > 1
        ]
        # Token-set unigrams (substring can false-hit e.g. "12" inside ranges).
        blob_token_set = set(blob_tokens)
        unigram = sum(1 for t in q_tokens if t in blob_token_set) / float(len(q_tokens))

        def _ngram_hits(n: int) -> float:
            if len(raw_tokens) < n or len(blob_tokens) < n:
                return 0.0
            q_ng = {
                tuple(raw_tokens[i : i + n]) for i in range(len(raw_tokens) - n + 1)
            }
            b_ng = {
                tuple(blob_tokens[i : i + n]) for i in range(len(blob_tokens) - n + 1)
            }
            if not q_ng:
                return 0.0
            return sum(1 for g in q_ng if g in b_ng) / float(len(q_ng))

        # Prefer multi-token cues ("years and older") over accidental "12 years"
        # inside "<12 years" after tokenization still matches bigrams — trigrams help.
        return unigram + 0.6 * _ngram_hits(2) + 1.1 * _ngram_hits(3)

    # Prefer tables whose surrounding text overlaps the question (indication/age cues).
    ranked_chunks = sorted(table_chunks, key=_chunk_overlap, reverse=True)

    verified: List[tuple] = []  # (overlap, VerifiedCell)
    last_insuff: Optional[InsufficientEvidence] = None
    for chunk in ranked_chunks:
        outcome = lookup_from_chunk(chunk, question)
        if outcome is None:
            continue
        if isinstance(outcome, VerifiedCell):
            verified.append((_chunk_overlap(chunk), outcome))
        elif isinstance(outcome, InsufficientEvidence):
            last_insuff = outcome

    if not verified:
        insuff = last_insuff or InsufficientEvidence(
            "no verified cell match in retrieved tables"
        )
        # Soft fail: allow prose RAG; avoid noisy abstain banners in answers.
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=insuff,
            prompt_prefix="",
            sources_update={
                "insufficient_table_evidence": True,
                "reason": insuff.reason,
                "soft_insufficient": True,
            },
        )

    # Prefer agreement across chunks; disagreeing values → abstain.
    values = {v.value for _, v in verified}
    if len(values) > 1:
        # If the top-overlap table is unique, trust it over lower-ranked conflicts.
        verified.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
        top_overlap, top_cell = verified[0]
        if top_overlap >= 0.2 and all(
            abs(top_overlap - ov) > 1e-9 or cell.value == top_cell.value
            for ov, cell in verified
            if ov >= top_overlap - 1e-9
        ):
            best = top_cell
        else:
            insuff = InsufficientEvidence(
                "retrieved tables disagree on the intersecting cell value"
            )
            return TableInterpretationResult(
                is_structured_lookup=True,
                result=insuff,
                prompt_prefix="",
                sources_update={
                    "insufficient_table_evidence": True,
                    "reason": insuff.reason,
                    "soft_insufficient": True,
                },
            )
    else:
        best = max(verified, key=lambda item: (item[0], item[1].confidence))[1]

    verified_meta = {
        "value": best.value,
        "row_headers": best.row_headers,
        "col_headers": best.col_headers,
        "page": best.page,
        "confidence": best.confidence,
        "evidence": dict(best.evidence or {}),
    }
    # Surface compare-mode so answer reconcile does not force takeaway = label cell.
    if (best.evidence or {}).get("prescribed_value") is not None:
        verified_meta["compare_mode"] = True
        verified_meta["prescribed_value"] = best.evidence.get("prescribed_value")
        verified_meta["prescribed_unit"] = best.evidence.get("prescribed_unit")

    return TableInterpretationResult(
        is_structured_lookup=True,
        result=best,
        prompt_prefix=best.to_prompt_block() + "\n\n",
        sources_update={"verified_cell": verified_meta},
    )
