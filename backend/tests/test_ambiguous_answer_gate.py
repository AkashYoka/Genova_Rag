"""Regression: ambiguous table evidence must not yield definitive PV conclusions."""
from __future__ import annotations

from app.rag.generation.structured_lookup import (
    STATUS_AMBIGUOUS,
    STATUS_EMPTY_CELL,
    STATUS_VERIFICATION_FAILED,
)
from app.rag.generation.table_selection import (
    applicability_dimensions,
    dimension_kind,
    select_eligible_table_chunks,
)
from app.rag.llm.agent import _reconcile_answer_with_verified_cell
from app.rag.table_rag.schema import TableLookupDecision


def test_dimension_kind_separates_provenance():
    assert dimension_kind("patient_age_group") == "applicability"
    assert dimension_kind("population") == "applicability"
    assert dimension_kind("trial_name") == "provenance"
    assert dimension_kind("section") == "provenance"
    assert dimension_kind("caption") == "provenance"
    assert applicability_dimensions(
        ["population", "trial_name", "section", "age_band", "caption"]
    ) == ["population", "age_band"]


def test_unitless_ae_table_excluded_from_multi_axis_lookup():
    """AE/narrative tables without axis units must not conflict with matrices."""
    dosing = {
        "table_id": "t_dose",
        "page": 5,
        "caption": "Table 1. Doses for Patients 12 Years and Older with Asthma",
        "headers": ["Assay (IU/mL)", "70-90 kg Dose (mg)"],
        "rows": [[">300-400", "300"]],
        "llm_understanding": {
            "applicability_cues": ["asthma", "12 years and older"],
            "applicability_attributes": [
                {"key": "indication", "value": "asthma"},
                {"key": "caption", "value": "Table 1. Doses..."},
                {"key": "section", "value": "Dosing Tables"},
            ],
            "differentiating_dimensions": [
                "indication",
                "caption",
                "section",
                "patient_age_group",
            ],
        },
    }
    ae = {
        "table_id": "t_ae",
        "page": 20,
        "caption": "Table 9. Adverse Reactions in Pediatric Patients",
        "headers": ["Adverse Reaction", "XOLAIR", "Placebo"],
        "rows": [["Headache", "6%", "4%"]],
        "llm_understanding": {
            "applicability_cues": ["pediatric", "adverse reactions"],
            "applicability_attributes": [
                {"key": "section", "value": "Adverse Reactions"},
                {"key": "caption", "value": "Table 9. Adverse..."},
            ],
            "differentiating_dimensions": ["patient_age_group", "section", "caption"],
        },
    }
    decision = select_eligible_table_chunks(
        "A 35-year-old with allergic asthma, 78 kg, IgE 350 IU/mL. "
        "What label dose from the table?",
        [
            {"table_id": "t_dose", "table": dosing, "caption": dosing["caption"]},
            {"table_id": "t_ae", "table": ae, "caption": ae["caption"]},
        ],
        get_table=lambda c: c["table"],
    )
    assert not decision.requires_disambiguation
    assert [c["table_id"] for c in decision.eligible_chunks] == ["t_dose"]


def test_clinical_takeaway_replaced_when_ambiguous():
    """Exact failure mode: definitive on-label conclusion + ambiguity footer."""
    sources = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_AMBIGUOUS,
            "missing_dimensions": ["patient_age_group"],
            "candidate_values": ["150", "300"],
            "candidate_table_ids": ["t_p5", "t_p7"],
            "reason": "ambiguous_applicability",
            "verification_status": "ambiguous",
            "selector_status": "needs_disambiguation",
        }
    ]
    answer = (
        "**Clinical Takeaway**\n"
        "For this patient the label recommends **150 mg every 4 weeks**.\n\n"
        "**Pharmacovigilance Classification**\n"
        "| **Off‑label Use** | **No** |\n"
        "The prescription is **on‑label** and appropriate.\n"
    )
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert out.startswith("Inconclusive")
    assert "Clinical Takeaway" not in out
    assert "Pharmacovigilance Classification" not in out
    assert "**No**" not in out
    assert "150 mg every 4 weeks" not in out
    assert "t_p5" in out
    assert "patient_age_group" in out


def test_ambiguous_after_selection_still_blocks_generation():
    sources = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_AMBIGUOUS,
            "candidate_values": ["225", "450"],
            "candidate_table_ids": ["t1", "t3"],
            "missing_dimensions": ["indication"],
            "verification_status": "ambiguous",
        }
    ]
    # Model ignored the abstain prompt and wrote a confident answer.
    answer = "Authoritative verified table value: 450 mg. On-label: Yes."
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert "Inconclusive" in out
    assert "450 mg" not in out or "Conflicting candidate values" in out
    assert "On-label: Yes" not in out


def test_verification_failed_blocks_definitive_answer():
    sources = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_VERIFICATION_FAILED,
            "reason": "selector row_index out of bounds",
            "candidate_table_ids": ["t1"],
            "verification_status": STATUS_VERIFICATION_FAILED,
        }
    ]
    answer = "**Clinical Takeaway** Label recommends 300 mg every 2 weeks. Off-label: No."
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert out.startswith("Inconclusive")
    assert "300 mg" not in out


def test_empty_cell_blocks_definitive_answer():
    sources = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_EMPTY_CELL,
            "reason": "matched cell is empty/unsupported",
            "verification_status": STATUS_EMPTY_CELL,
            "candidate_table_ids": ["t1"],
        }
    ]
    answer = "The labeled dose is 225 mg every 4 weeks (on-label)."
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert "Inconclusive" in out
    assert "225 mg every 4 weeks" not in out


def test_unique_verified_still_allows_definitive_pin():
    sources = [
        {
            "verified_cell": {
                "value": "300",
                "status": "matched",
                "page": 5,
                "table_id": "t_p5",
                "row_headers": [">300-400"],
                "col_headers": [">70-90 kg"],
            }
        }
    ]
    answer = "The recommended dose is 150 mg every 4 weeks."
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert "300" in out
    assert out.startswith("**Authoritative verified table value: 300**") or "300" in out


def test_provenance_dims_alone_do_not_force_ambiguity():
    """section/trial_name must not block a uniquely applicable dosing table."""
    adult = {
        "table_id": "t_adult",
        "page": 5,
        "caption": "Table 1. Doses for Patients 12 Years and Older with Asthma",
        "headers": ["Assay (IU/mL)", "70-90 kg"],
        "rows": [[">300-400", "300"]],
        "llm_understanding": {
            "purpose": "asthma dosing adults",
            "applicability_cues": ["asthma", "12 years and older"],
            "applicability_attributes": [
                {"key": "indication", "value": "asthma"},
                {"key": "section", "value": "Dosing Tables"},
            ],
            "differentiating_dimensions": [
                "indication",
                "patient_age_group",
                "section",
                "trial_name",
            ],
        },
    }
    other = {
        "table_id": "t_other",
        "page": 7,
        "caption": "Table 3. Doses for Adult Patients with CRSwNP",
        "headers": ["Assay (IU/mL)", "70-90 kg"],
        "rows": [[">300-400", "450"]],
        "llm_understanding": {
            "purpose": "CRSwNP dosing",
            "applicability_cues": ["CRSwNP"],
            "applicability_attributes": [
                {"key": "indication", "value": "CRSwNP"},
                {"key": "section", "value": "Dosing Tables"},
            ],
            "differentiating_dimensions": ["indication", "section", "trial_name"],
        },
    }
    chunks = [
        {"table_id": "t_adult", "table": adult, "caption": adult["caption"], "score": 0.5},
        {"table_id": "t_other", "table": other, "caption": other["caption"], "score": 0.9},
    ]
    decision = select_eligible_table_chunks(
        "A 35-year-old with allergic asthma, 78 kg, IgE 350 IU/mL. "
        "What label dose from the table?",
        chunks,
        get_table=lambda c: c["table"],
    )
    assert not decision.requires_disambiguation
    assert len(decision.eligible_chunks) == 1
    assert decision.eligible_chunks[0]["table_id"] == "t_adult"


def test_selector_decision_accepts_null_lists():
    d = TableLookupDecision.model_validate(
        {
            "status": "matched",
            "table_id": "t1",
            "row_index": 0,
            "col_index": 1,
            "missing_dimensions": None,
            "operands": None,
            "operation": None,
            "rationale": "ok",
        }
    )
    assert d.missing_dimensions == []
    assert d.operands == []
    assert d.operation == ""
