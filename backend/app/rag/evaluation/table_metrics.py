"""Deterministic table-specific evaluation metrics (no fabricated scores)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence


_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _nums(text: str) -> List[float]:
    out: List[float] = []
    for m in _NUM_RE.finditer(str(text or "").replace(",", "")):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def numeric_answer_match(reference: Any, answer: Any, *, tol: float = 1e-6) -> Optional[bool]:
    """Compare numeric answers when both sides yield parseable numbers."""
    if reference is None or answer is None:
        return None
    ref_nums = _nums(str(reference))
    ans_nums = _nums(str(answer))
    if not ref_nums or not ans_nums:
        # Fall back to substring when non-numeric reference.
        ref = _norm(reference)
        act = _norm(answer)
        if not ref:
            return None
        return ref in act
    return any(abs(a - r) <= tol for r in ref_nums for a in ans_nums)


def unit_preserved(reference: Any, answer: Any) -> Optional[bool]:
    """True when a unit token from the reference appears in the answer."""
    if reference is None or answer is None:
        return None
    units = re.findall(
        r"(?i)\b(mg|kg|g|ml|l|%|iu/?mL|rs\.?|₹|\$)\b", str(reference)
    )
    if not units:
        return None
    ans = str(answer).lower()
    return any(u.lower().replace(" ", "") in ans.replace(" ", "") for u in units)


def page_attribution_ok(
    expected_page: Any,
    sources: Sequence[Dict[str, Any]],
) -> Optional[bool]:
    if expected_page is None:
        return None
    try:
        want = int(expected_page)
    except (TypeError, ValueError):
        return None
    pages: List[int] = []
    for src in sources or []:
        for key in ("page", "page_start"):
            try:
                pages.append(int(src.get(key)))
            except (TypeError, ValueError):
                pass
        verified = src.get("verified_cell") or {}
        if isinstance(verified, dict) and verified.get("page") is not None:
            try:
                pages.append(int(verified["page"]))
            except (TypeError, ValueError):
                pass
    if not pages:
        return False
    return want in pages


def table_id_selected(
    expected_table: Any,
    sources: Sequence[Dict[str, Any]],
    *,
    chunks: Sequence[Dict[str, Any]] | None = None,
) -> Optional[bool]:
    if not expected_table:
        return None
    want = str(expected_table).strip()
    ids: List[str] = []
    for src in sources or []:
        if src.get("table_id"):
            ids.append(str(src["table_id"]))
        verified = src.get("verified_cell") or {}
        if isinstance(verified, dict) and verified.get("table_id"):
            ids.append(str(verified["table_id"]))
    for chunk in chunks or []:
        if chunk.get("table_id"):
            ids.append(str(chunk["table_id"]))
    if not ids:
        return False
    return any(want in i or i in want for i in ids)


def evaluate_table_case(
    case: Dict[str, Any],
    *,
    answer: str,
    sources: Sequence[Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]] | None = None,
    verified: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute deterministic table metrics; unset fields stay null (not invented)."""
    reference = case.get("reference") or case.get("expected_value")
    verified_value = None
    if verified and verified.get("value") is not None:
        verified_value = verified.get("value")

    return {
        "numeric_answer_correct": numeric_answer_match(reference, answer),
        "verified_cell_correct": (
            numeric_answer_match(reference, verified_value)
            if reference is not None and verified_value is not None
            else None
        ),
        "unit_preserved": unit_preserved(reference, answer),
        "page_attribution_ok": page_attribution_ok(case.get("expected_page"), sources),
        "table_id_selected": table_id_selected(
            case.get("expected_table") or case.get("expected_table_id"),
            sources,
            chunks=chunks,
        ),
        "selector_status": next(
            (
                str(s.get("selector_status"))
                for s in sources
                if s.get("selector_status")
            ),
            None,
        ),
        "has_verified_cell": bool(verified),
        "tags": list(case.get("tags") or []),
    }
