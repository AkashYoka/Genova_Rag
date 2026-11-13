"""Geometry-aware header reconstruction + multi-domain structured lookup."""
from __future__ import annotations

import json

from app.rag.generation.structured_lookup import (
    STATUS_AMBIGUOUS,
    STATUS_MATCHED,
    STATUS_OUT_OF_RANGE,
    lookup_cell_in_table,
    parse_range_label,
)
from app.rag.generation.table_interpretation import interpret_table_question
from app.rag.ingestion.pdf_layout.header_reconstruction import (
    reconstruct_table_headers,
)


def test_multi_level_header_stitch_body_weight():
    header_rows = [
        ["Assay", "Freq", "", "Body", "Weight", ""],
        ["", "", "30-60 kg", ">60-70 kg", ">70-90 kg", ">90-150 kg"],
    ]
    rows = [
        [">300-400", "Every", "225", "225", "300", "375"],
        [">400-500", "2", "300", "300", "375", ""],
        [">500-600", "weeks", "300", "375", "Insufficient", "Data"],
    ]
    merged = [
        {"row_index": 0, "col_index": 2, "row_span": 1, "col_span": 4, "text": ""},
    ]
    recon = reconstruct_table_headers(
        header_rows=header_rows,
        rows=rows,
        merged_cells=merged,
        existing_headers=["Assay", "Freq", "30-60 kg", ">60-70 kg", ">70-90 kg", ">90-150 kg"],
    )
    assert recon.leaf_headers[4] == ">70-90 kg"
    # Body + Weight should be stitched into hierarchy path for weight cols.
    path4 = recon.normalized_header_paths[4]
    assert any("Body" in p and "Weight" in p for p in path4) or any(
        p == "Body Weight" for p in path4
    )
    # Frequency column reconstructed from fragments.
    assert "Every" in recon.reconstructed_rows[0][1]
    assert "4" in recon.reconstructed_rows[0][1] or "2" in recon.reconstructed_rows[0][1]
    assert recon.normalization_confidence >= 0.5


def test_promote_misclassified_leaf_header_band():
    recon = reconstruct_table_headers(
        header_rows=[
            [
                "Assay",
                "Freq",
                "Body Weight",
                "Body Weight",
                "Body Weight",
                "Body Weight",
            ]
        ],
        rows=[
            ["", "", "30-60 kg", ">60-70 kg", ">70-90 kg", ">90-150 kg"],
            [">300-400", "Every 2 weeks", "225", "225", "300", "375"],
        ],
        existing_headers=["Assay", "Freq", "Body Weight", "Body Weight", "Body Weight", "Body Weight"],
    )
    assert recon.leaf_headers[4] == ">70-90 kg"
    assert "promoted_body_row_to_header_continuation" in recon.warnings
    assert recon.reconstructed_rows[0][0] == ">300-400"


def test_space_separated_range_repair_and_parse():
    recon = reconstruct_table_headers(
        header_rows=[["Assay", "40-50 kg"]],
        rows=[[">300 400", "600"]],
        existing_headers=["Assay", "40-50 kg"],
    )
    assert recon.reconstructed_rows[0][0] == ">300-400"
    parsed = parse_range_label(">300 400")
    assert parsed is not None
    assert parsed.low == 300
    assert parsed.high == 400


def test_healthcare_fixture_lookup_matched():
    table = {
        "table_id": "hc_adult",
        "page": 5,
        "caption": "Table 1. Doses for Patients 12 Years of Age and Older with Asthma",
        "headers": ["Assay (IU/mL)", "Schedule", "30-60 kg", ">60-70 kg", ">70-90 kg"],
        "normalized_header_paths": [
            ["Assay (IU/mL)"],
            ["Schedule"],
            ["Body Weight", "30-60 kg"],
            ["Body Weight", ">60-70 kg"],
            ["Body Weight", ">70-90 kg"],
        ],
        "rows": [
            [">300-400", "Every 2 weeks", "225", "300", "300"],
        ],
        "llm_understanding": {
            "applicability_cues": ["12 years of age and older", "Asthma"],
            "applicability_attributes": [
                {
                    "key": "age_band",
                    "value": "≥12 years",
                    "evidence_type": "explicit",
                    "confidence": 0.9,
                    "provenance": "caption",
                }
            ],
            "differentiating_dimensions": ["age_band", "indication"],
        },
    }
    result = lookup_cell_in_table(
        table,
        "For a 45-year-old with assay 350 IU/mL and body weight 78 kg, what value?",
    )
    assert result.status == STATUS_MATCHED
    assert result.value == "300"
    assert result.evidence["normalized_header_paths"]["column"][-1] == ">70-90 kg"


def test_finance_domain_period_ambiguity():
    """Finance matrices distinguished by reporting_period (not healthcare fields)."""
    from app.rag.generation.table_selection import select_eligible_table_chunks

    fy2023 = {
        "table_id": "fin_2023",
        "page": 1,
        "caption": "Margin by load band — FY2023",
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
    fy2024 = {
        "table_id": "fin_2024",
        "page": 2,
        "caption": "Margin by load band — FY2024",
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

    def chunk(t):
        return {
            "table_id": t["table_id"],
            "page": t["page"],
            "chunk_type": "table",
            "section": t["caption"],
            "text": t["caption"],
            "filename": "finance.pdf",
            "table": t,
            "table_json": json.dumps(t),
        }

    # No period → both eligible / requires disambiguation.
    decision = select_eligible_table_chunks(
        "What margin applies at 35% load and 25 t from the table?",
        [chunk(fy2023), chunk(fy2024)],
    )
    assert decision.requires_disambiguation is True

    # FY2024 cue uniquely selects one table.
    decision2 = select_eligible_table_chunks(
        "What margin applies at 35% load and 25 t in FY2024 from the table?",
        [chunk(fy2023), chunk(fy2024)],
    )
    assert len(decision2.eligible_chunks) == 1
    assert decision2.eligible_chunks[0]["table_id"] == "fin_2024"

    result = interpret_table_question(
        "What margin applies at 35% load and 25 t in FY2024 from the table?",
        [chunk(fy2023), chunk(fy2024)],
    )
    assert result.verified is not None
    assert result.verified.value == "16"


def test_engineering_domain_material_grades():
    grade_a = {
        "table_id": "eng_a36",
        "page": 3,
        "caption": "Allowable stress — ASTM A36",
        "headers": ["Thickness (mm)", "Stress (MPa)"],
        "rows": [["10-20", "150"], ["20-40", "140"]],
        "llm_understanding": {
            "applicability_cues": ["ASTM A36"],
            "applicability_attributes": [
                {"key": "material_grade", "value": "ASTM A36"}
            ],
            "differentiating_dimensions": ["material_grade"],
        },
    }
    grade_b = {
        "table_id": "eng_a572",
        "page": 4,
        "caption": "Allowable stress — ASTM A572",
        "headers": ["Thickness (mm)", "Stress (MPa)"],
        "rows": [["10-20", "200"], ["20-40", "190"]],
        "llm_understanding": {
            "applicability_cues": ["ASTM A572"],
            "applicability_attributes": [
                {"key": "material_grade", "value": "ASTM A572"}
            ],
            "differentiating_dimensions": ["material_grade"],
        },
    }

    def chunk(t):
        return {
            "table_id": t["table_id"],
            "page": t["page"],
            "chunk_type": "table",
            "section": t["caption"],
            "text": t["caption"],
            "filename": "eng.pdf",
            "table": t,
            "table_json": json.dumps(t),
        }

    q = "From the stress table, what stress applies at 15 mm thickness for ASTM A36?"
    result = interpret_table_question(q, [chunk(grade_a), chunk(grade_b)])
    assert result.verified is not None
    assert result.verified.value == "150"
    assert result.verified.table_id == "eng_a36"


def test_out_of_range_and_missing_applicability_not_wildcard():
    adult = {
        "table_id": "t_adult",
        "page": 5,
        "caption": "Patients 12 years of age and older",
        "headers": ["Assay", "70-90 kg"],
        "rows": [[">900-1000", "Insufficient Data"]],
        "llm_understanding": {
            "applicability_cues": ["12 years of age and older"],
            "applicability_attributes": [{"key": "age_band", "value": "≥12 years"}],
            "differentiating_dimensions": ["age_band"],
        },
    }
    result = lookup_cell_in_table(
        adult,
        "For a 45-year-old assay 950 IU/mL and body weight 80 kg what value from the table?",
    )
    assert result.status == STATUS_OUT_OF_RANGE
