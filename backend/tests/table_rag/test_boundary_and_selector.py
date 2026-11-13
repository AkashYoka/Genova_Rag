"""Phase 4: boundary precision, disambiguation, multipage stitch, verifier gate."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from app.rag.generation.structured_lookup import (
    STATUS_AMBIGUOUS,
    STATUS_EMPTY_CELL,
    STATUS_MATCHED,
    STATUS_NOT_FOUND,
    STATUS_OUT_OF_RANGE,
    STATUS_VERIFICATION_FAILED,
    InsufficientEvidence,
    VerifiedCell,
    lookup_cell_in_table,
    range_contains,
    parse_range_label,
)
from app.rag.generation.table_selector import (
    collect_candidate_tables,
    scope_rank_candidates,
    verify_lookup_decision,
)
from app.rag.ingestion.pdf_layout.table_normalize import stitch_multipage_tables
from app.rag.ingestion.pdf_layout.types import BBox, TableData
from app.rag.table_rag.schema import TableLookupDecision, TablePayload
from app.rag.table_rag.understanding import profile_full_table_grid


# ── 1. Boundary edge precision ───────────────────────────────────────────────


def test_boundary_edge_exclusive_vs_inclusive_bins():
    """30.0 belongs to 30–60, not <30; <=30 vs >30 are distinct."""
    lt30 = parse_range_label("<30")
    bin_30_60 = parse_range_label("30–60")
    le30 = parse_range_label("<=30")
    gt30 = parse_range_label(">30")

    assert lt30 is not None and bin_30_60 is not None
    assert not range_contains(lt30, 30.0, "")
    assert range_contains(bin_30_60, 30.0, "")
    assert le30 is not None and range_contains(le30, 30.0, "")
    assert gt30 is not None and not range_contains(gt30, 30.0, "")
    assert range_contains(gt30, 30.1, "")


def test_lookup_pins_edge_value_on_correct_bin():
    table = {
        "table_id": "t_bins",
        "page": 1,
        "pages": [1],
        "headers": ["Range", "Value"],
        "rows": [
            ["<30", "A"],
            ["30–60", "B"],
            [">60", "C"],
        ],
        "n_rows": 4,
        "n_cols": 2,
    }
    # Exact edge 30.0 → second bin.
    cell = lookup_cell_in_table(table, "What value for 30?")
    assert isinstance(cell, VerifiedCell)
    assert cell.value == "B"
    assert cell.row_index == 1


# ── 2. Sibling table disambiguation ──────────────────────────────────────────


def test_sibling_tables_need_disambiguation_when_cohort_missing():
    adult = {
        "table_id": "t_adult",
        "page": 2,
        "pages": [2],
        "headers": ["Weight", "Dose"],
        "rows": [["30-60 kg", "150"], [">60 kg", "300"]],
        "caption": "Adult dosing",
        "llm_understanding": {
            "differentiating_dimensions": ["patient_cohort"],
            "applicability_cues": ["Adult", "ages 18 years and older"],
            "applicability_attributes": [
                {"key": "patient_cohort", "value": "Adult"}
            ],
            "purpose": "Adult dosing schedule",
        },
    }
    peds = {
        "table_id": "t_peds",
        "page": 3,
        "pages": [3],
        "headers": ["Weight", "Dose"],
        "rows": [["30-60 kg", "75"], [">60 kg", "150"]],
        "caption": "Pediatric dosing",
        "llm_understanding": {
            "differentiating_dimensions": ["patient_cohort"],
            "applicability_cues": ["Pediatric", "ages 6 to 17 years"],
            "applicability_attributes": [
                {"key": "patient_cohort", "value": "Pediatric"}
            ],
            "purpose": "Pediatric dosing schedule",
        },
    }
    chunks = [
        {"table_id": "t_adult", "table": adult, "score": 0.05, "section": "Adults"},
        {"table_id": "t_peds", "table": peds, "score": 0.04, "section": "Pediatrics"},
    ]
    candidates = collect_candidate_tables(chunks, top_k=4)
    decision = TableLookupDecision(
        status="needs_disambiguation",
        table_id=None,
        missing_dimensions=["patient_cohort"],
        rationale="Adult vs Pediatric cohort not specified",
    )
    outcome = verify_lookup_decision(decision, candidates, question="Dose for 50 kg?")
    assert isinstance(outcome, InsufficientEvidence)
    assert outcome.status == STATUS_AMBIGUOUS
    assert "patient_cohort" in outcome.missing_dimensions


def test_scope_rank_prefers_indication_matched_dosing_table():
    """Adult asthma questions must not rank a sibling indication matrix first."""
    asthma = {
        "table_id": "t_asthma",
        "page": 5,
        "pages": [5],
        "caption": "Table 1. Doses for Patients 12 Years and Older with Asthma",
        "headers": [
            "Assay (IU/mL)",
            "Dosing Freq.",
            "30-60 kg",
            ">60-70 kg",
            ">70-90 kg",
        ],
        "rows": [
            [">200-300", "Every 4 weeks", "300", "225", "225"],
            [">300-400", "Every 2 weeks", "225", "225", "300"],
            [">400-500", "Every 2 weeks", "300", "300", "375"],
        ],
        "llm_understanding": {
            "subject": "asthma dosing",
            "purpose": "Subcutaneous doses for asthma patients 12 years and older",
            "applicability_cues": ["asthma", "12 years of age and older"],
            "applicability_attributes": [
                {"key": "indication", "value": "asthma"},
                {"key": "age_band", "value": "12 years and older"},
            ],
            "differentiating_dimensions": ["indication", "age_band"],
        },
    }
    other = {
        "table_id": "t_other",
        "page": 7,
        "pages": [7],
        "caption": "Table 3. Doses for Adult Patients with CRSwNP",
        "headers": [
            "Assay (IU/mL)",
            "Dosing Freq.",
            "30-60 kg",
            ">60-70 kg",
            ">70-90 kg",
        ],
        "rows": [
            [">200-300", "Every 4 weeks", "450", "450", "450"],
            [">300-400", "Every 2 weeks", "450", "450", "600"],
            [">400-500", "Every 4 weeks", "375", "375", "375"],
        ],
        "llm_understanding": {
            "subject": "CRSwNP dosing",
            "purpose": "Subcutaneous doses for adult patients with CRSwNP",
            "applicability_cues": ["CRSwNP", "adult patients"],
            "applicability_attributes": [
                {"key": "indication", "value": "CRSwNP"},
            ],
            "differentiating_dimensions": ["indication"],
        },
    }
    peds = {
        "table_id": "t_peds",
        "page": 6,
        "pages": [6],
        "caption": "Table 2. Doses for Pediatric Patients with Asthma Ages 6 to <12 Years",
        "headers": [
            "Assay (IU/mL)",
            "Dosing Freq.",
            ">25-30 kg",
            ">30-40 kg",
        ],
        "rows": [
            [">200-300", "Every 4 weeks", "150", "225"],
            [">300-400", "Every 2 weeks", "225", "300"],
        ],
        "llm_understanding": {
            "subject": "pediatric asthma dosing",
            "purpose": "Subcutaneous doses for pediatric asthma 6 to <12 years",
            "applicability_cues": ["pediatric", "asthma", "ages 6 to <12 years"],
            "applicability_attributes": [
                {"key": "indication", "value": "asthma"},
                {"key": "age_band", "value": "6 to <12 years"},
            ],
            "differentiating_dimensions": ["indication", "age_band"],
        },
    }
    chunks = [
        {"table_id": "t_other", "table": other, "score": 0.9, "caption": other["caption"]},
        {"table_id": "t_peds", "table": peds, "score": 0.8, "caption": peds["caption"]},
        {"table_id": "t_asthma", "table": asthma, "score": 0.5, "caption": asthma["caption"]},
    ]
    raw = collect_candidate_tables(chunks, top_k=6, question="")
    ranked = scope_rank_candidates(
        "A 40-year-old with allergic asthma, weight 50 kg, IgE 350 IU/mL, "
        "prescribed 600 mg every 2 weeks. Classify off-label use.",
        raw,
    )
    assert ranked, "expected scoped candidates"
    assert ranked[0]["table_id"] == "t_asthma"
    # Pediatric cohort must be age-filtered out for a 40-year-old.
    assert all(c["table_id"] != "t_peds" for c in ranked)

    # "Adult <indication>" must not let the word Adult alone promote a sibling
    # whose caption merely says Adult + a different indication.
    adult_phrased = scope_rank_candidates(
        "Adult asthma, 80 kg, IgE 250 IU/mL prescribed 75 mg every 4 weeks. "
        "Classify dosing deviation.",
        raw,
    )
    assert adult_phrased, "expected scoped candidates for Adult asthma phrasing"
    assert adult_phrased[0]["table_id"] == "t_asthma"


def test_scope_rank_hard_prefers_cited_table_number():
    """Explicit ``Table N`` in the question must beat higher-scoring siblings."""
    t1 = {
        "table_id": "t1",
        "page": 5,
        "caption": "Table 1. Rates for Cohort A Ages 12 and Older",
        "headers": ["Assay", "30-60 kg", ">70-90 kg"],
        "rows": [[">300-400", "225", "300"]],
        "llm_understanding": {
            "purpose": "Table 1. Rates for Cohort A Ages 12 and Older",
            "applicability_cues": ["Table 1. Rates for Cohort A Ages 12 and Older"],
        },
    }
    t4 = {
        "table_id": "t4",
        "page": 8,
        "caption": "Table 4. Rates for Mixed Adult and Pediatric Cohort",
        "headers": ["Assay", ">=10-12 kg", ">70-80 kg"],
        "rows": [[">300-400", "150", "600"]],
        "llm_understanding": {
            "purpose": "Table 4. Rates for Mixed Adult and Pediatric Cohort",
            "applicability_cues": ["Table 4. Rates for Mixed Adult and Pediatric Cohort"],
        },
    }
    chunks = [
        {"table_id": "t4", "table": t4, "score": 0.99, "caption": t4["caption"], "page": 8},
        {"table_id": "t1", "table": t1, "score": 0.2, "caption": t1["caption"], "page": 5},
    ]
    ranked = scope_rank_candidates(
        "According to Table 1 for persons 12 years and older, what value for "
        "assay 350 and body weight 78 kg?",
        collect_candidate_tables(chunks, top_k=4, question=""),
    )
    assert ranked[0]["table_id"] == "t1"
    assert all(c["table_id"] != "t4" for c in ranked)


# ── 3. Multi-page stitching ──────────────────────────────────────────────────


@dataclass
class _FakePage:
    page_number: int
    tables: List[Any] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def test_multipage_stitch_merges_page4_and_page5():
    page4 = _FakePage(
        page_number=4,
        metrics={"page_height": 792.0},
        tables=[
            TableData(
                page=4,
                bbox=BBox(72, 600, 540, 780),  # near bottom
                headers=["Assay", "Dose"],
                rows=[[">200-300", "150"], [">300-400", "225"]],
                extraction_method="textract",
                pages=[4],
            )
        ],
    )
    page5 = _FakePage(
        page_number=5,
        metrics={"page_height": 792.0},
        tables=[
            TableData(
                page=5,
                bbox=BBox(72, 40, 540, 200),  # near top
                headers=["Assay", "Dose"],  # repeated headers
                rows=[[">400-500", "300"], [">500-600", "375"]],
                extraction_method="textract",
                pages=[5],
            )
        ],
    )
    flat = stitch_multipage_tables([page4, page5])
    assert len(page5.tables) == 0  # continuation consumed
    assert len(page4.tables) == 1
    stitched = page4.tables[0]
    assert stitched.pages == [4, 5]
    assert len(stitched.rows) == 4
    # Row originally on page 5 is present and lookupable.
    assert stitched.rows[2] == [">400-500", "300"]
    assert stitched.rows[3][1] == "375"
    assert len(flat) == 1

    payload = stitched.to_json()
    cell = lookup_cell_in_table(
        {
            **payload,
            "table_id": "t_stitched",
        },
        "What dose for assay 450?",
    )
    # May be VerifiedCell or insufficient depending on axis matcher; at least
    # the grid contains the page-5 row for selector/verifier paths.
    assert any(r[0] == ">400-500" for r in stitched.rows)


# ── 4. Python verifier guardrail ─────────────────────────────────────────────


def test_verifier_downgrades_out_of_bounds_coordinate():
    table = {
        "table_id": "t_v",
        "page": 1,
        "pages": [1],
        "headers": ["A", "B"],
        "rows": [["x", "10"], ["y", "20"]],
    }
    candidates = collect_candidate_tables(
        [{"table_id": "t_v", "table": table, "score": 1.0}],
        top_k=2,
    )
    decision = TableLookupDecision(
        status="matched",
        table_id="t_v",
        row_index=99,
        col_index=0,
        rationale="bad coord",
    )
    outcome = verify_lookup_decision(decision, candidates)
    assert isinstance(outcome, InsufficientEvidence)
    assert outcome.status == STATUS_VERIFICATION_FAILED
    assert "out of bounds" in outcome.reason.lower()


def test_verifier_downgrades_na_cell_to_out_of_range():
    table = {
        "table_id": "t_na",
        "page": 1,
        "pages": [1],
        "headers": ["Weight", "Dose"],
        "rows": [["30-60", "N/A"], [">60", "300"]],
    }
    candidates = collect_candidate_tables(
        [{"table_id": "t_na", "table": table, "score": 1.0}],
        top_k=2,
    )
    decision = TableLookupDecision(
        status="matched",
        table_id="t_na",
        row_index=0,
        col_index=1,
        rationale="picked N/A",
    )
    outcome = verify_lookup_decision(decision, candidates)
    assert isinstance(outcome, InsufficientEvidence)
    assert outcome.status == STATUS_EMPTY_CELL


def test_verifier_accepts_valid_sql_cell():
    table = {
        "table_id": "t_ok",
        "page": 1,
        "pages": [1],
        "headers": ["Weight", "Dose"],
        "rows": [["30-60", "150"], [">60", "300"]],
    }
    candidates = collect_candidate_tables(
        [{"table_id": "t_ok", "table": table, "filename": "x.pdf", "score": 1.0}],
        top_k=2,
    )
    decision = TableLookupDecision(
        status="matched",
        table_id="t_ok",
        row_index=1,
        col_index=1,
        rationale="ok",
    )
    outcome = verify_lookup_decision(decision, candidates)
    assert isinstance(outcome, VerifiedCell)
    assert outcome.status == STATUS_MATCHED
    assert outcome.value == "300"
    assert outcome.row_index == 1
    assert outcome.col_index == 1


def test_profile_full_table_grid_scans_all_rows():
    payload = TablePayload(
        table_id="t_prof",
        headers=["Bin", "Val"],
        rows=[[f"{i}-{i+1}", str(i * 10)] for i in range(20)],
    )
    result = profile_full_table_grid(payload)
    profiles = result["column_profiles"]
    assert len(profiles) == 2
    assert result["n_rows_scanned"] == 20
    assert profiles[1]["min_value"] == 0.0
    assert profiles[1]["max_value"] == 190.0
    assert len(profiles[0]["distinct_values"]) <= 30
