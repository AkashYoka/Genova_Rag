"""Answer-stage verified table evidence helpers."""
from __future__ import annotations

from app.rag.generation.table_evidence import VERIFIED_BLOCK_MARKER, VerifiedCell
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


def test_reconcile_rewrites_wrong_dose_unit():
    """Dose cells must keep Dose(mg); never borrow IgE IU or invent mcg."""
    sources = [
        {
            "verified_cell": {
                "value": "300",
                "row_headers": [">300-400 IU/mL"],
                "col_headers": [">70-90 kg", "Dose (mg)"],
                "table_headers": [
                    "Pretreatment Serum IgE (IU/mL)",
                    "Dosing Freq.",
                    "Dose (mg)",
                ],
                "page": 5,
                "status": "matched",
                "evidence": {"unit": "mg"},
            }
        }
    ]
    answer = (
        "Takeaway: the recommended XOLAIR dose is **300 IU** every 2 weeks "
        "[Source: xolair_prescribing.pdf, Page 5]."
    )
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert "300 mg" in out.lower() or "300 **mg" in out.lower()
    assert "300 iu" not in out.lower().replace("iu/ml", "")


def test_reconcile_rewrites_microgram_dose_unit():
    sources = [
        {
            "verified_cell": {
                "value": "300",
                "row_headers": [">300-400"],
                "col_headers": [">70-90 kg"],
                "page": 5,
                "status": "matched",
                "evidence": {"unit": "mg"},
            }
        }
    ]
    answer = "The recommended XOLAIR dose is **300 µg** every 2 weeks."
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert "300 mg" in out.lower()
    assert "µg" not in out.lower() and "μg" not in out.lower()


def test_reconcile_rewrites_dose_claimed_as_iu_per_ml():
    sources = [
        {
            "verified_cell": {
                "value": "300",
                "row_headers": [">300-400"],
                "col_headers": [">70-90 kg"],
                "page": 5,
                "status": "matched",
                "evidence": {"unit": "mg"},
            }
        }
    ]
    answer = (
        "The recommended dose is **300 IU/mL every 2 weeks**. "
        "Pretreatment IgE is 350 IU/mL."
    )
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert "300 mg" in out.lower()
    assert "350 iu/ml" in out.lower()


def test_reconcile_skips_pin_on_wrong_table_narrative():
    sources = [
        {
            "verified_cell": {
                "value": "300",
                "row_headers": [">200-300"],
                "col_headers": ["30-60 kg", "Dose (mg)"],
                "page": 5,
                "status": "matched",
                "evidence": {"unit": "mg"},
            }
        }
    ]
    answer = (
        "The adult Table 1 regimen is the wrong table for this pediatric patient. "
        "Using 300 mg every 2 weeks from the adult matrix is not appropriate."
    )
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert "Authoritative verified table value" not in out
