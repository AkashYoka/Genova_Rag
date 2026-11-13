"""Domain-agnostic table selection + multi-table conflict regression tests."""
from __future__ import annotations

import json

from app.rag.generation.structured_lookup import (
    STATUS_AMBIGUOUS,
    STATUS_MATCHED,
    STATUS_NOT_FOUND,
    STATUS_OUT_OF_RANGE,
    lookup_cell_in_table,
)
from app.rag.generation.table_interpretation import interpret_table_question
from app.rag.generation.table_selection import (
    build_table_applicability,
    select_eligible_table_chunks,
)
from app.rag.llm.agent import _reconcile_answer_with_verified_cell


def _adult_matrix() -> dict:
    return {
        "table_id": "t_adult",
        "page": 5,
        "caption": "Patients 12 years of age and older",
        "headers": [
            "Assay (IU/mL)",
            "Schedule",
            "30-60 kg",
            ">60-70 kg",
            ">70-90 kg",
        ],
        "rows": [
            [">200-300", "Every 4 weeks", "150", "225", "300"],
            [">300-400", "Every 2 weeks", "225", "300", "300"],
        ],
        "llm_understanding": {
            "subject": "lookup grid ages 12 years and older",
            "purpose": "recommended value by assay and body weight",
            "applicability_cues": [
                "12 years of age and older",
                "≥12 years",
            ],
            "applicability_attributes": [
                {"key": "age_band", "value": "≥12 years"}
            ],
            "differentiating_dimensions": ["age_band"],
            "semantic_search_phrases": ["ages 12 years and older"],
        },
    }


def _pediatric_matrix() -> dict:
    return {
        "table_id": "t_peds",
        "page": 7,
        "caption": "Patients 6 to <12 years of age",
        "headers": [
            "Assay (IU/mL)",
            "Schedule",
            ">60-70 kg",
            ">70-80 kg",
            ">80-90 kg",
        ],
        "rows": [
            [">200-300", "Every 2 weeks", "225", "300", "375"],
            [">300-400", "Every 2 weeks", "300", "600", "600"],
        ],
        "llm_understanding": {
            "subject": "lookup grid ages 6 to <12 years",
            "purpose": "recommended value by assay and body weight",
            "applicability_cues": [
                "6 to <12 years of age",
                "ages 6 to <12 years",
            ],
            "applicability_attributes": [
                {"key": "age_band", "value": "6 to <12 years"}
            ],
            "differentiating_dimensions": ["age_band"],
            "semantic_search_phrases": ["ages 6 to <12 years"],
        },
    }


def _finance_matrix() -> dict:
    return {
        "table_id": "t_fy2024",
        "page": 2,
        "caption": "Revenue by segment — FY2024",
        "headers": ["Segment", "Q1", "Q2", "Q3", "Q4"],
        "rows": [
            ["Cloud", "10", "12", "14", "16"],
            ["Hardware", "8", "7", "9", "11"],
        ],
        "llm_understanding": {
            "subject": "revenue FY2024",
            "applicability_cues": ["FY2024", "fiscal year 2024"],
            "applicability_attributes": [
                {"key": "reporting_period", "value": "FY2024"}
            ],
            "differentiating_dimensions": ["reporting_period"],
            "semantic_search_phrases": ["FY2024 revenue by segment"],
        },
    }


def _chunk(table: dict, *, text: str = "") -> dict:
    return {
        "table_id": table["table_id"],
        "page": table["page"],
        "chunk_type": "table",
        "section": table.get("caption") or "",
        "text": text or table.get("caption") or "table",
        "filename": "fixture.pdf",
        "table": table,
        "table_json": json.dumps(table),
    }


def test_conflicting_sibling_tables_without_age_are_ambiguous():
    """Regression: disagreeing matrices must not be resolved by overlap."""
    q = (
        "For assay 350 IU/mL and body weight 78 kg, what is the recommended value "
        "from the dosing table?"
    )
    chunks = [_chunk(_adult_matrix()), _chunk(_pediatric_matrix())]
    result = interpret_table_question(q, chunks)
    assert result.insufficient is not None
    assert result.insufficient.status == STATUS_AMBIGUOUS
    assert result.verified is None
    assert "300" in result.insufficient.candidate_values
    assert "600" in result.insufficient.candidate_values
    assert result.sources_update.get("soft_insufficient") is False
    assert "AMBIGUOUS" in result.prompt_prefix


def test_age_constraint_selects_adult_matrix_only():
    q = (
        "For a 45-year-old patient with assay 350 IU/mL and body weight 78 kg, "
        "what value applies according to the table?"
    )
    chunks = [_chunk(_adult_matrix()), _chunk(_pediatric_matrix())]
    result = interpret_table_question(q, chunks)
    assert result.verified is not None
    assert result.verified.status == STATUS_MATCHED
    assert result.verified.value == "300"
    assert result.verified.table_id == "t_adult"
    assert result.verified.page == 5


def test_age_constraint_selects_pediatric_matrix_only():
    q = (
        "For an 8-year-old patient with assay 350 IU/mL and body weight 78 kg, "
        "what value applies according to the table?"
    )
    chunks = [_chunk(_adult_matrix()), _chunk(_pediatric_matrix())]
    result = interpret_table_question(q, chunks)
    assert result.verified is not None
    assert result.verified.value == "600"
    assert result.verified.table_id == "t_peds"


def test_reconcile_does_not_pin_on_ambiguous():
    sources = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_AMBIGUOUS,
            "candidate_values": ["300", "600"],
            "missing_dimensions": ["age_band"],
        }
    ]
    answer = "The recommended dose is 600 mg every 2 weeks."
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert out.startswith("Ambiguous table evidence")
    assert "600" in out  # original prose retained after banner


def test_reconcile_pins_provenance_when_answer_cites_other_page():
    sources = [
        {
            "verified_cell": {
                "value": "225",
                "status": "matched",
                "page": 5,
                "row_headers": [">200-300"],
                "col_headers": [">60-70 kg"],
                "table_id": "t_adult",
            }
        }
    ]
    answer = (
        "The recommended value is 225 every 4 weeks "
        "[Source: doc.pdf, Page 7] from a different section."
    )
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert out.startswith("**Authoritative verified table value: 225**")
    assert "Page 5" in out.split("\n\n", 1)[0]


def test_verified_compare_prompt_marks_equal_amount_as_match():
    from app.rag.generation.structured_lookup import VerifiedCell

    cell = VerifiedCell(
        value="225",
        row_headers=[">300-400"],
        col_headers=[">60-70 kg"],
        row_index=1,
        col_index=3,
        page=5,
        table_id="t_a",
        evidence={
            "prescribed_value": 225.0,
            "prescribed_unit": "mg",
            "table_caption": "Patients 12 years and older",
        },
    )
    block = cell.to_prompt_block()
    assert "EQUALS the verified label cell" in block
    assert "Do NOT invent a deviation solely because a stated amount was mentioned" in block
    assert "Page 5" in block
    assert "cite THIS page and table only" in block


def test_soft_insufficient_prompt_allows_prose_fallback():
    from app.rag.generation.structured_lookup import InsufficientEvidence

    soft = InsufficientEvidence(
        "no row or column labels matched question values",
        status=STATUS_NOT_FOUND,
        soft=True,
    )
    block = soft.to_prompt_block()
    assert "SOFT table-lookup notice" in block
    assert "Do not invent a matrix cell" in block
    hard = InsufficientEvidence(
        "multiple applicable tables",
        status=STATUS_AMBIGUOUS,
        soft=False,
        candidate_values=["225", "450"],
    )
    assert "do not invent a single cell value" in hard.to_prompt_block().lower()


def test_out_of_range_unsupported_cell():
    table = _adult_matrix()
    table["rows"] = [[">300-400", "Every 2 weeks", "225", "300", "Insufficient Data"]]
    result = lookup_cell_in_table(
        table,
        "What value for assay 350 IU/mL and body weight 78 kg?",
    )
    assert hasattr(result, "status")
    assert result.status == STATUS_OUT_OF_RANGE


def test_finance_fixture_period_cues_are_domain_agnostic():
    table = _finance_matrix()
    profile = build_table_applicability(table)
    assert any("fy2024" in c.lower() for c in profile.cues)
    assert profile.attributes.get("reporting_period") == "FY2024"
    decision = select_eligible_table_chunks(
        "What was Cloud Q4 revenue in FY2024?",
        [_chunk(table)],
    )
    assert decision.eligible_chunks
    # Single table — no forced ambiguity.
    assert decision.requires_disambiguation is False


def test_xolair_style_expected_adult_cell_from_fixture_not_extraction():
    """Expected values authored in the fixture (not from live Textract)."""
    expected_value = "300"
    expected_page = 5
    expected_table = "t_adult"
    q = (
        "For a 45-year-old with pretreatment assay 350 IU/mL and body weight "
        "78 kg, what is the recommended table value?"
    )
    result = interpret_table_question(
        q, [_chunk(_adult_matrix()), _chunk(_pediatric_matrix())]
    )
    assert result.verified is not None
    assert result.verified.value == expected_value
    assert result.verified.page == expected_page
    assert result.verified.table_id == expected_table


def _class_matrix(table_id: str, page: int, class_value: str, cell: str) -> dict:
    """Synthetic sibling grids that differ only by a categorical attribute."""
    return {
        "table_id": table_id,
        "page": page,
        "caption": f"Lookup grid — class {class_value}",
        "headers": [
            "Assay (IU/mL)",
            "Schedule",
            "30-60 kg",
            ">60-70 kg",
            ">70-90 kg",
        ],
        "rows": [
            [">200-300", "Every 4 weeks", "150", "225", cell],
            [">300-400", "Every 2 weeks", "225", "300", cell],
        ],
        "llm_understanding": {
            "subject": f"lookup grid for {class_value}",
            "purpose": "recommended value by assay and body weight",
            "applicability_cues": [class_value, f"class {class_value}"],
            "applicability_attributes": [
                {"key": "applicability_class", "value": class_value},
            ],
            "differentiating_dimensions": ["applicability_class"],
            "semantic_search_phrases": [f"{class_value} lookup grid"],
        },
    }


def _segment_matrix(table_id: str, page: int, segment: str, cell: str) -> dict:
    """Finance-style sibling grids (proves no medical coupling)."""
    return {
        "table_id": table_id,
        "page": page,
        "caption": f"Revenue — {segment}",
        "headers": ["Metric", "Q1", "Q2", "Q3", "Q4"],
        "rows": [
            ["Cloud", "10", "12", "14", cell],
        ],
        "llm_understanding": {
            "subject": f"revenue {segment}",
            "applicability_cues": [segment, f"segment {segment}"],
            "applicability_attributes": [
                {"key": "reporting_segment", "value": segment},
            ],
            "differentiating_dimensions": ["reporting_segment"],
            "semantic_search_phrases": [f"{segment} revenue"],
        },
    }


def test_categorical_class_token_selects_matching_sibling():
    """Query token matching an attribute VALUE must pin that table, not abstain."""
    class_a = _class_matrix("t_class_a", 1, "alpha", "300")
    class_b = _class_matrix("t_class_b", 2, "bravo", "600")
    q = (
        "For alpha with assay 350 IU/mL and body weight 78 kg, "
        "what is the recommended value from the table?"
    )
    result = interpret_table_question(q, [_chunk(class_a), _chunk(class_b)])
    assert result.verified is not None
    assert result.verified.status == STATUS_MATCHED
    assert result.verified.value == "300"
    assert result.verified.table_id == "t_class_a"


def test_categorical_class_omitted_stays_ambiguous_when_values_disagree():
    """Without the differentiating value, disagreeing siblings stay ambiguous."""
    class_a = _class_matrix("t_class_a", 1, "alpha", "300")
    class_b = _class_matrix("t_class_b", 2, "bravo", "600")
    q = (
        "For assay 350 IU/mL and body weight 78 kg, what is the recommended "
        "value from the dosing table?"
    )
    result = interpret_table_question(q, [_chunk(class_a), _chunk(class_b)])
    assert result.insufficient is not None
    assert result.insufficient.status == STATUS_AMBIGUOUS
    assert result.verified is None
    assert "300" in result.insufficient.candidate_values
    assert "600" in result.insufficient.candidate_values


def test_age_plus_class_token_still_selects_correct_grid():
    """Age bounds must not skip categorical coverage when a class token is present."""
    adult_a = _adult_matrix()
    adult_a["table_id"] = "t_adult_a"
    adult_a["llm_understanding"] = {
        **adult_a["llm_understanding"],
        "applicability_cues": [
            "12 years of age and older",
            "≥12 years",
            "alpha",
        ],
        "applicability_attributes": [
            {"key": "age_band", "value": "≥12 years"},
            {"key": "applicability_class", "value": "alpha"},
        ],
        "differentiating_dimensions": ["age_band", "applicability_class"],
    }
    adult_b = _adult_matrix()
    adult_b["table_id"] = "t_adult_b"
    adult_b["page"] = 6
    adult_b["rows"] = [
        [">200-300", "Every 4 weeks", "150", "225", "450"],
        [">300-400", "Every 2 weeks", "225", "300", "600"],
    ]
    adult_b["llm_understanding"] = {
        **adult_b["llm_understanding"],
        "applicability_cues": [
            "12 years of age and older",
            "≥12 years",
            "bravo",
        ],
        "applicability_attributes": [
            {"key": "age_band", "value": "≥12 years"},
            {"key": "applicability_class", "value": "bravo"},
        ],
        "differentiating_dimensions": ["age_band", "applicability_class"],
    }
    q = (
        "For a 45-year-old alpha case with assay 350 IU/mL and body weight "
        "78 kg, what value applies according to the table?"
    )
    result = interpret_table_question(q, [_chunk(adult_a), _chunk(adult_b)])
    assert result.verified is not None
    assert result.verified.value == "300"
    assert result.verified.table_id == "t_adult_a"


def test_finance_segment_token_selects_matching_sibling():
    retail = _segment_matrix("t_retail", 1, "retail", "16")
    wholesale = _segment_matrix("t_wholesale", 2, "wholesale", "22")
    decision = select_eligible_table_chunks(
        "What was Cloud Q4 revenue for the retail segment?",
        [_chunk(retail), _chunk(wholesale)],
    )
    assert len(decision.eligible_chunks) == 1
    assert decision.eligible_chunks[0]["table_id"] == "t_retail"
    assert decision.requires_disambiguation is False
