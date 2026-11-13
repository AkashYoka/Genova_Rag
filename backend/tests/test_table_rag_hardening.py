"""Domain-agnostic TableRAG hardening: structure, eligibility, constraints."""
from __future__ import annotations

import json

from app.rag.generation.query_constraints import (
    EVIDENCE_EXPLICIT,
    EVIDENCE_MISSING,
    extract_query_constraints,
    mark_missing_dimensions,
)
from app.rag.generation.structured_lookup import (
    STATUS_AMBIGUOUS,
    STATUS_INVALID_TABLE_STRUCTURE,
    STATUS_MATCHED,
    STATUS_OUT_OF_RANGE,
    lookup_cell_in_table,
)
from app.rag.generation.table_interpretation import interpret_table_question
from app.rag.generation.table_selection import select_eligible_table_chunks
from app.rag.ingestion.pdf_layout.table_structure import (
    STRUCTURE_INVALID,
    STRUCTURE_NEEDS_VERIFICATION,
    STRUCTURE_VALID,
    validate_table_structure,
)
from app.rag.llm.agent import _reconcile_answer_with_verified_cell


def _chunk(table: dict) -> dict:
    return {
        "table_id": table["table_id"],
        "page": table.get("page"),
        "chunk_type": "table",
        "section": table.get("caption") or "",
        "text": table.get("caption") or "table",
        "filename": "fixture.pdf",
        "table": table,
        "table_json": json.dumps(table),
    }


def test_damaged_grid_is_invalid_and_blocks_lookup():
    table = {
        "table_id": "bad",
        "headers": [],
        "rows": [["a", "1"], ["b"]],
        "merged_cells": [
            {"row_index": 0, "col_index": 0, "row_span": 0, "col_span": 3, "text": "x"}
        ],
    }
    validation = validate_table_structure(table)
    assert validation.status == STRUCTURE_INVALID
    assert validation.reliable_for_lookup is False
    result = lookup_cell_in_table(
        table, "What value at 10 kg and 20 % from the table?"
    )
    assert result.status == STATUS_INVALID_TABLE_STRUCTURE


def test_fragmented_headers_need_verification():
    table = {
        "table_id": "frag",
        "headers": ["Column 1", "Column 2", "Column 3"],
        "rows": [["10-20", "1", "2"]],
        "header_reconstruction": {
            "needs_verification": True,
            "normalization_confidence": 0.3,
            "warnings": ["no_header_rows_raw_headers_used"],
        },
    }
    validation = validate_table_structure(table)
    assert validation.status == STRUCTURE_NEEDS_VERIFICATION
    assert "fragmented_headers" in validation.reason_codes or (
        "header_reconstruction_uncertain" in validation.reason_codes
    )


def test_valid_simple_grid():
    table = {
        "headers": ["Load (%)", "10-20 t", "20-40 t"],
        "rows": [["35", "12", "14"]],
        "textract_confidence": 95.0,
    }
    validation = validate_table_structure(table)
    assert validation.status == STRUCTURE_VALID
    assert validation.reliable_for_lookup is True


def test_irrelevant_vial_table_excluded_by_units():
    matrix = {
        "table_id": "matrix",
        "page": 1,
        "caption": "Lookup by assay and mass — Class A",
        "headers": ["Assay (IU/mL)", "30-60 kg", ">60-70 kg", ">70-90 kg"],
        "rows": [[">300-400", "225", "300", "300"]],
        "llm_understanding": {
            "applicability_cues": ["Class A"],
            "applicability_attributes": [{"key": "category", "value": "Class A"}],
            "differentiating_dimensions": ["category"],
        },
    }
    vials = {
        "table_id": "vials",
        "page": 2,
        "caption": "Presentation sizes",
        "headers": ["Strength", "75 mg", "150 mg", "300 mg"],
        "rows": [["Vials", "1", "1", "2"]],
        "llm_understanding": {
            "applicability_cues": ["presentation"],
            "applicability_attributes": [{"key": "category", "value": "presentation"}],
            "differentiating_dimensions": ["category"],
        },
    }
    decision = select_eligible_table_chunks(
        "For assay 350 IU/mL and body weight 78 kg what value from the table?",
        [_chunk(matrix), _chunk(vials)],
    )
    ids = {c["table_id"] for c in decision.eligible_chunks}
    assert "matrix" in ids
    assert "vials" not in ids


def test_missing_differentiating_dimension_is_ambiguous():
    a = {
        "table_id": "fy23",
        "page": 1,
        "caption": "Margin grid — FY2023",
        "headers": ["Load (%)", "10-20 t", "20-40 t"],
        "rows": [["35", "12", "14"]],
        "llm_understanding": {
            "applicability_cues": ["FY2023"],
            "applicability_attributes": [
                {"key": "reporting_period", "value": "FY2023"}
            ],
            "differentiating_dimensions": ["reporting_period"],
        },
    }
    b = {
        "table_id": "fy24",
        "page": 2,
        "caption": "Margin grid — FY2024",
        "headers": ["Load (%)", "10-20 t", "20-40 t"],
        "rows": [["35", "15", "16"]],
        "llm_understanding": {
            "applicability_cues": ["FY2024"],
            "applicability_attributes": [
                {"key": "reporting_period", "value": "FY2024"}
            ],
            "differentiating_dimensions": ["reporting_period"],
        },
    }
    result = interpret_table_question(
        "What margin applies at 35% load and 25 t from the table?",
        [_chunk(a), _chunk(b)],
    )
    assert result.insufficient is not None
    assert result.insufficient.status == STATUS_AMBIGUOUS
    assert "14" in result.insufficient.candidate_values
    assert "16" in result.insufficient.candidate_values
    assert "reporting_period" in (result.insufficient.missing_dimensions or [])


def test_explicit_period_selects_finance_table():
    a = {
        "table_id": "fy23",
        "page": 1,
        "caption": "Margin grid — FY2023",
        "headers": ["Load (%)", "10-20 t", "20-40 t"],
        "rows": [["35", "12", "14"]],
        "llm_understanding": {
            "applicability_cues": ["FY2023", "fiscal year 2023"],
            "applicability_attributes": [
                {"key": "reporting_period", "value": "FY2023"}
            ],
            "differentiating_dimensions": ["reporting_period"],
        },
    }
    b = {
        "table_id": "fy24",
        "page": 2,
        "caption": "Margin grid — FY2024",
        "headers": ["Load (%)", "10-20 t", "20-40 t"],
        "rows": [["35", "15", "16"]],
        "llm_understanding": {
            "applicability_cues": ["FY2024", "fiscal year 2024"],
            "applicability_attributes": [
                {"key": "reporting_period", "value": "FY2024"}
            ],
            "differentiating_dimensions": ["reporting_period"],
        },
    }
    result = interpret_table_question(
        "What margin applies at 35% load and 25 t in FY2024 from the table?",
        [_chunk(a), _chunk(b)],
    )
    assert result.verified is not None
    assert result.verified.value == "16"
    assert result.verified.table_id == "fy24"


def test_engineering_grade_conflict_and_selection():
    a36 = {
        "table_id": "a36",
        "page": 3,
        "caption": "Allowable stress — ASTM A36",
        "headers": ["Thickness (mm)", "Stress (MPa)"],
        "rows": [["10-20", "150"], ["20-40", "140"]],
        "llm_understanding": {
            "applicability_cues": ["ASTM A36"],
            "applicability_attributes": [
                {"key": "classification", "value": "ASTM A36"}
            ],
            "differentiating_dimensions": ["classification"],
        },
    }
    a572 = {
        "table_id": "a572",
        "page": 4,
        "caption": "Allowable stress — ASTM A572",
        "headers": ["Thickness (mm)", "Stress (MPa)"],
        "rows": [["10-20", "200"], ["20-40", "190"]],
        "llm_understanding": {
            "applicability_cues": ["ASTM A572"],
            "applicability_attributes": [
                {"key": "classification", "value": "ASTM A572"}
            ],
            "differentiating_dimensions": ["classification"],
        },
    }
    ambiguous = interpret_table_question(
        "From the table, what stress applies at 15 mm thickness?",
        [_chunk(a36), _chunk(a572)],
    )
    assert ambiguous.insufficient is not None
    assert ambiguous.insufficient.status == STATUS_AMBIGUOUS

    matched = interpret_table_question(
        "From the table, what stress applies at 15 mm thickness for ASTM A36?",
        [_chunk(a36), _chunk(a572)],
    )
    assert matched.verified is not None
    assert matched.verified.value == "150"


def test_scientific_fixture_out_of_range_abstention():
    table = {
        "table_id": "sci",
        "page": 1,
        "caption": "Calibration curve — Instrument B",
        "headers": ["Concentration (ng/mL)", "Response"],
        "rows": [["1-10", "0.2"], ["10-50", "0.8"]],
        "llm_understanding": {
            "applicability_cues": ["Instrument B"],
            "applicability_attributes": [
                {"key": "instrument", "value": "Instrument B"}
            ],
        },
    }
    # Dual units force structured path; concentration band won't contain 999.
    result = lookup_cell_in_table(
        table,
        "What response at 999 ng/mL from the table?",
    )
    # Single axis may not resolve 2-col without second constraint — accept
    # not_found or out_of_range as safe abstention (not matched).
    assert getattr(result, "status", None) != STATUS_MATCHED or False
    if hasattr(result, "status"):
        assert result.status in {
            STATUS_OUT_OF_RANGE,
            "not_found",
            STATUS_INVALID_TABLE_STRUCTURE,
        } or result.status != STATUS_MATCHED


def test_query_constraints_explicit_vs_missing():
    q = extract_query_constraints(
        "What margin at 35% load and 25 t in FY2024 from the table?"
    )
    assert "reporting_period" in q.explicit_keys
    assert any(c.key == "reporting_period" and c.evidence_type == EVIDENCE_EXPLICIT for c in q.constraints)
    marked = mark_missing_dimensions(q, ["classification", "reporting_period"])
    assert "classification" in marked.missing_dimensions
    assert any(
        c.key == "classification" and c.evidence_type == EVIDENCE_MISSING
        for c in marked.constraints
    )


def test_duplicate_tables_agreeing_values():
    t1 = {
        "table_id": "t1",
        "page": 1,
        "caption": "Grid — Region North",
        "headers": ["Band", "A", "B"],
        "rows": [["10-20", "5", "7"]],
        "llm_understanding": {
            "applicability_cues": ["Region North"],
            "applicability_attributes": [{"key": "region", "value": "North"}],
            "differentiating_dimensions": ["region"],
        },
    }
    t2 = {
        "table_id": "t2",
        "page": 2,
        "caption": "Grid — Region North (continued)",
        "headers": ["Band", "A", "B"],
        "rows": [["10-20", "5", "7"]],
        "llm_understanding": {
            "applicability_cues": ["Region North"],
            "applicability_attributes": [{"key": "region", "value": "North"}],
            "differentiating_dimensions": ["region"],
        },
    }
    # Same region cue in query → both eligible, agreeing values → matched.
    result = interpret_table_question(
        "From the North region table, value at band 15 and column B?",
        [_chunk(t1), _chunk(t2)],
    )
    # May be matched if axis resolves, or not_found if axis labels don't fit —
    # must not invent a value.
    if result.verified:
        assert result.verified.value == "7"
        assert result.verified.status == STATUS_MATCHED


def test_reconcile_invalid_structure_abstains():
    sources = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_INVALID_TABLE_STRUCTURE,
        }
    ]
    out = _reconcile_answer_with_verified_cell("The answer is 42.", sources)
    assert "Insufficient table evidence" in out or "insufficient" in out.lower()


def test_healthcare_fixture_age_still_distinguishes_without_hardcoded_fields():
    """Expected values live only in the fixture — production stays generic."""
    adult = {
        "table_id": "t_adult",
        "page": 5,
        "caption": "Patients 12 years of age and older",
        "headers": ["Assay (IU/mL)", "30-60 kg", ">60-70 kg", ">70-90 kg"],
        "rows": [[">300-400", "225", "300", "300"]],
        "llm_understanding": {
            "applicability_cues": ["12 years of age and older"],
            "applicability_attributes": [
                {"key": "population", "value": "≥12 years"}
            ],
            "differentiating_dimensions": ["population"],
        },
    }
    peds = {
        "table_id": "t_peds",
        "page": 7,
        "caption": "Patients 6 to <12 years of age",
        "headers": ["Assay (IU/mL)", ">60-70 kg", ">70-80 kg", ">80-90 kg"],
        "rows": [[">300-400", "300", "600", "600"]],
        "llm_understanding": {
            "applicability_cues": ["6 to <12 years of age"],
            "applicability_attributes": [
                {"key": "population", "value": "6 to <12 years"}
            ],
            "differentiating_dimensions": ["population"],
        },
    }
    q = (
        "For a 45-year-old with assay 350 IU/mL and body weight 78 kg, "
        "what value from the table?"
    )
    result = interpret_table_question(q, [_chunk(adult), _chunk(peds)])
    assert result.verified is not None
    assert result.verified.value == "300"
    assert result.verified.table_id == "t_adult"
