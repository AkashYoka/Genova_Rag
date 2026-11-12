"""Answer-stage verified table evidence helpers."""
from __future__ import annotations

from app.rag.generation.structured_lookup import VERIFIED_BLOCK_MARKER, VerifiedCell
from app.rag.llm.agent import _reconcile_answer_with_verified_cell


def test_apply_verified_evidence_prefix_compare_mode():
    cell = VerifiedCell(
        value="300",
        row_headers=[">300-400"],
        col_headers=[">70-90 kg"],
        row_index=1,
        col_index=4,
        page=3,
        filename="doc.pdf",
        evidence={
            "match_mode": "axis_lookup_with_prescribed_compare",
            "prescribed_value": 150.0,
            "prescribed_unit": "mg",
        },
    )
    block = cell.to_prompt_block()
    assert VERIFIED_BLOCK_MARKER in block
    assert "150" in block
    assert "MUST state this value verbatim" not in block


def test_reconcile_compare_mode_keeps_stated_amount():
    sources = [
        {
            "verified_cell": {
                "value": "300",
                "row_headers": [">300-400"],
                "col_headers": [">70-90 kg"],
                "page": 3,
                "compare_mode": True,
                "prescribed_value": 150.0,
                "prescribed_unit": "mg",
            }
        }
    ]
    answer = "Takeaway: prescribed **150 mg** is below the label cell of 300."
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert out == answer


def test_reconcile_compare_mode_injects_label_if_missing():
    sources = [
        {
            "verified_cell": {
                "value": "300",
                "row_headers": [">300-400"],
                "col_headers": [">70-90 kg"],
                "page": 3,
                "compare_mode": True,
                "prescribed_value": 150.0,
                "prescribed_unit": "mg",
            }
        }
    ]
    answer = "Takeaway: the prescribed 150 mg every 4 weeks looks fine."
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert out.startswith("**Verified label table value: 300**")
    assert "150 mg" in out
