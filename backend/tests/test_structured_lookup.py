"""Domain-agnostic structured table lookup tests."""
from __future__ import annotations

from app.rag.generation.structured_lookup import (
    VERIFIED_BLOCK_MARKER,
    extract_numeric_constraints,
    lookup_cell_in_table,
    ROLE_AXIS,
    ROLE_COMPARE,
    ROLE_EXCLUDE,
    ROLE_FREQUENCY,
)
from app.rag.generation.table_interpretation import interpret_table_question
from app.rag.llm.agent import (
    _apply_table_interpretation,
    _reconcile_answer_with_verified_cell,
)


def _matrix_table() -> dict:
    # One row per assay band (typical matrix). Duplicate schedule siblings for
    # the same assay band are covered in a separate ambiguity test.
    return {
        "page": 3,
        "headers": [
            "Assay (IU/mL)",
            "Schedule",
            "30-60 kg",
            ">60-70 kg",
            ">70-90 kg",
        ],
        "rows": [
            [">200-300", "Every 4 weeks", "150", "225", "300"],
            [">300-400", "Every 2 weeks", "225", "225", "300"],
            ["≥30-100", "Every 4 weeks", "150", "150", "150"],
        ],
    }


def test_extract_roles_age_weight_assay_prescribed():
    q = (
        "A 35-year-old subject, body weight 78 kg and pretreatment assay "
        "350 IU/mL was intentionally prescribed 150 mg every 4 weeks."
    )
    cs = extract_numeric_constraints(q)
    roles = {(c.value, c.unit or "", c.role) for c in cs}
    assert (35.0, "", ROLE_EXCLUDE) in roles
    assert (78.0, "kg", ROLE_AXIS) in roles
    assert (350.0, "iu/ml", ROLE_AXIS) in roles
    assert (150.0, "mg", ROLE_COMPARE) in roles
    assert any(c.role == ROLE_FREQUENCY and c.value == 4.0 for c in cs)


def test_age_does_not_match_assay_band():
    table = _matrix_table()
    q = (
        "A 35-year-old subject, body weight 78 kg and pretreatment assay "
        "350 IU/mL was intentionally prescribed 150 mg every 4 weeks."
    )
    result = lookup_cell_in_table(table, q, filename="doc.pdf", page=3)
    assert hasattr(result, "value")
    assert result.value == "300"
    assert ">300-400" in " ".join(result.row_headers)
    assert ">70-90" in " ".join(result.col_headers)
    assert result.evidence.get("prescribed_value") == 150.0
    assert "≥30-100" not in " ".join(result.row_headers)


def test_pure_axis_lookup():
    table = _matrix_table()
    q = "What value applies for assay 350 IU/mL and body weight 65 kg every 2 weeks?"
    result = lookup_cell_in_table(table, q, page=3)
    assert hasattr(result, "value")
    assert result.value == "225"
    assert result.evidence.get("prescribed_value") is None


def test_compare_prompt_does_not_force_takeaway_equals_label():
    table = _matrix_table()
    q = (
        "A 35-year-old subject, body weight 78 kg and pretreatment assay "
        "350 IU/mL was intentionally prescribed 150 mg every 4 weeks. "
        "Identify any dosing discrepancy."
    )
    chunk = {
        "filename": "doc.pdf",
        "page": 3,
        "chunk_type": "table",
        "text": "table",
        "table_json": table,
        "section": "Dosing",
    }
    interp = interpret_table_question(q, [chunk])
    assert interp.verified is not None
    assert VERIFIED_BLOCK_MARKER in interp.prompt_prefix
    assert (
        "prescribed" in interp.prompt_prefix.lower()
        or "administered" in interp.prompt_prefix.lower()
        or "stated amount" in interp.prompt_prefix.lower()
    )
    assert "MUST state this value verbatim: 300" not in interp.prompt_prefix

    ctx, sources = _apply_table_interpretation(q, [chunk], "chunk body", [])
    assert VERIFIED_BLOCK_MARKER in ctx
    answer = (
        "### Takeaway\nThe prescribed 150 mg every 4 weeks is not label-concordant; "
        "the table cell is 300 mg every 2 weeks."
    )
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert not out.startswith("**Authoritative verified table value:")
    assert "150" in out
    assert "300" in out


def test_reconcile_pure_lookup_still_pins():
    sources = [
        {
            "verified_cell": {
                "value": "300",
                "row_headers": [">300-400"],
                "col_headers": [">70-90 kg"],
                "page": 3,
            }
        }
    ]
    answer = "### Takeaway\nThe applicable value is **150 mg** every 4 weeks."
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert out.startswith("**Authoritative verified table value: 300**")
