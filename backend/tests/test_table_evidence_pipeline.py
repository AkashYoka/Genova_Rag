"""Domain-neutral regressions for the table evidence pipeline.

Covers: applicability vs provenance, explicit eligibility states, generalized
numeric matching, provenance records, and the global answer-consistency gate
(non-streaming + streaming). All fixtures are synthetic and industry-neutral.
"""
from __future__ import annotations

import pytest

from app.rag.generation.structured_lookup import (
    ANSWER_STATE_AMBIGUOUS,
    ANSWER_STATE_INSUFFICIENT,
    ANSWER_STATE_INVALID,
    ANSWER_STATE_NOT_APPLICABLE,
    ANSWER_STATE_VERIFIED,
    STATUS_AMBIGUOUS,
    STATUS_INVALID_TABLE_STRUCTURE,
    VerifiedCell,
    answer_state_from_sources,
    lookup_cell_in_table,
    parse_range_label,
    range_contains,
)
from app.rag.generation.table_interpretation import interpret_table_question
from app.rag.generation.table_selection import (
    ELIGIBLE,
    INELIGIBLE,
    UNCERTAIN,
    dimension_kind,
    profiles_are_distinct,
    select_eligible_table_chunks,
)
from app.rag.llm.agent import (
    _reconcile_answer_with_verified_cell,
    _streaming_gate_message,
)
from app.rag.table_rag.telemetry import (
    begin_table_rag_trace,
    clear_table_rag_trace,
    current_table_rag_trace,
    record_candidate_audit,
    record_cell_verification,
    record_answer_state,
)


# ── Fixtures (synthetic, domain-neutral) ────────────────────────────────────


def _sibling(table_id: str, page: int, cohort: str, cell: str) -> dict:
    caption = f"Lookup grid for {cohort} cohort"
    return {
        "table_id": table_id,
        "page": page,
        "caption": caption,
        "headers": ["Assay (IU/mL)", "30-60 kg", ">60-70 kg"],
        "rows": [[">200-300", "150", "225"], [">300-400", "225", cell]],
        "llm_understanding": {
            "purpose": caption,
            "applicability_cues": [cohort, "assay and body weight"],
            "applicability_attributes": [{"key": "population", "value": cohort}],
            "differentiating_dimensions": ["population"],
        },
    }


def _period_table(
    table_id: str,
    page: int,
    caption: str,
    trial: str,
    value: str,
) -> dict:
    return {
        "table_id": table_id,
        "page": page,
        "caption": caption,
        "headers": ["Assay (IU/mL)", "30-60 kg", ">60-70 kg"],
        "rows": [[">300-400", "225", value]],
        "llm_understanding": {
            "purpose": "Lookup grid for reporting period FY2024",
            "applicability_cues": ["FY2024", "assay and body weight", "lookup grid"],
            "applicability_attributes": [
                {"key": "reporting_period", "value": "FY2024"},
                {"key": "trial_name", "value": trial},
            ],
            "differentiating_dimensions": ["reporting_period", "trial_name"],
        },
    }


def _bare_table(table_id: str, page: int, cell: str) -> dict:
    """No applicability attributes beyond layout provenance (section)."""
    return {
        "table_id": table_id,
        "page": page,
        "caption": "Lookup grid",
        "headers": ["Assay (IU/mL)", "30-60 kg", ">60-70 kg"],
        "rows": [[">300-400", "225", cell]],
        "llm_understanding": {
            "applicability_cues": ["lookup grid"],
            "applicability_attributes": [{"key": "section", "value": "Lookup"}],
            "differentiating_dimensions": ["section"],
        },
    }


def _chunk(table: dict, *, score: float = 0.5) -> dict:
    return {
        "table_id": table["table_id"],
        "page": table.get("page"),
        "caption": table.get("caption") or "",
        "chunk_type": "table",
        "section": table.get("caption") or "",
        "text": table.get("caption") or "table",
        "filename": "fixture.pdf",
        "table": table,
        "score": score,
    }


AXIS_QUERY = "What value applies for assay 350 IU/mL and body weight 65 kg?"


# 1. One clearly applicable table.
def test_single_clearly_applicable_table_is_verified():
    table = _sibling("t_one", 4, "AlphaCohort", "300")
    decision = select_eligible_table_chunks(
        AXIS_QUERY, [_chunk(table)], get_table=lambda c: c["table"]
    )
    assert [c["table_id"] for c in decision.eligible_chunks] == ["t_one"]
    assert decision.requires_disambiguation is False
    state = {c.table_id: c.state for c in decision.candidates}
    assert state["t_one"] == ELIGIBLE
    assert decision.candidates[0].sufficient_evidence is True

    result = interpret_table_question(AXIS_QUERY, [_chunk(table)])
    assert result.verified is not None
    assert result.verified.value == "300"


# 2. Multiple sibling tables with different populations.
def test_sibling_populations_are_ambiguous_without_population():
    tables = [
        _sibling("t_a", 1, "AlphaCohort", "300"),
        _sibling("t_b", 2, "BetaCohort", "450"),
    ]
    chunks = [_chunk(t) for t in tables]
    decision = select_eligible_table_chunks(
        AXIS_QUERY, chunks, get_table=lambda c: c["table"]
    )
    assert decision.requires_disambiguation is True
    assert "population" in " ".join(decision.missing_dimensions)
    states = {c.table_id: c for c in decision.candidates}
    assert states["t_a"].state == ELIGIBLE and states["t_b"].state == ELIGIBLE
    assert states["t_a"].population == "AlphaCohort"
    assert set(states["t_a"].conflicts_with) == {"t_b"}
    assert states["t_a"].provenance["page"] == "1"

    result = interpret_table_question(AXIS_QUERY, chunks)
    assert result.verified is None
    assert result.insufficient.status == STATUS_AMBIGUOUS


# 3. Identical numerical axes but different applicability.
def test_identical_axes_query_population_picks_one_table():
    tables = [
        _sibling("t_a", 1, "AlphaCohort", "300"),
        _sibling("t_b", 2, "BetaCohort", "450"),
    ]
    query = (
        "For the BetaCohort, what value applies for assay 350 IU/mL "
        "and body weight 65 kg?"
    )
    result = interpret_table_question(query, [_chunk(t) for t in tables])
    assert result.verified is not None
    assert result.verified.table_id == "t_b"
    assert result.verified.value == "450"


# 4. Different captions but identical applicability.
def test_different_captions_same_applicability_do_not_conflict():
    left = _period_table("t_1", 3, "Lookup grid — FY2024", "TrialOne", "300")
    right = _period_table(
        "t_2", 4, "Lookup grid — FY2024 (continued)", "TrialOne", "300"
    )
    decision = select_eligible_table_chunks(
        AXIS_QUERY, [_chunk(left), _chunk(right)], get_table=lambda c: c["table"]
    )
    assert decision.requires_disambiguation is False
    profiles = {
        c.table_id: c for c in decision.candidates
    }
    assert profiles["t_1"].state == ELIGIBLE
    assert profiles["t_1"].conflicts_with == []
    result = interpret_table_question(AXIS_QUERY, [_chunk(left), _chunk(right)])
    assert result.verified is not None
    assert result.verified.value == "300"


# 5. Different trial names without actual applicability conflicts.
def test_trial_name_difference_is_provenance_not_ambiguity():
    left = _period_table("t_1", 3, "Lookup grid — FY2024", "TrialOne", "300")
    right = _period_table("t_2", 4, "Lookup grid — FY2024", "TrialTwo", "300")
    assert dimension_kind("trial_name") == "provenance"
    decision = select_eligible_table_chunks(
        AXIS_QUERY, [_chunk(left), _chunk(right)], get_table=lambda c: c["table"]
    )
    assert decision.requires_disambiguation is False
    from app.rag.generation.table_selection import build_table_applicability

    assert not profiles_are_distinct(
        build_table_applicability(left), build_table_applicability(right)
    )


# 6. Missing age / population metadata is flagged, never silently eligible.
def test_missing_applicability_metadata_is_uncertain_not_eligible():
    tables = [_bare_table("t_1", 1, "300"), _bare_table("t_2", 2, "450")]
    decision = select_eligible_table_chunks(
        AXIS_QUERY, [_chunk(t) for t in tables], get_table=lambda c: c["table"]
    )
    states = {c.table_id: c for c in decision.candidates}
    assert states["t_1"].state == UNCERTAIN
    assert states["t_1"].sufficient_evidence is False
    assert "no_applicability_metadata" in states["t_1"].reasons or (
        "metadata_without_applicability_attributes" in states["t_1"].reasons
    )
    # Still retrievable (no over-abstention on a single candidate).
    single = select_eligible_table_chunks(
        AXIS_QUERY, [_chunk(tables[0])], get_table=lambda c: c["table"]
    )
    assert single.requires_disambiguation is False
    assert single.eligible_chunks


# 7. Missing or incompatible units.
def test_unitless_sibling_is_ineligible_for_multi_axis_lookup():
    matrix = _sibling("t_matrix", 1, "AlphaCohort", "300")
    unitless = {
        "table_id": "t_list",
        "page": 9,
        "caption": "Event summary",
        "headers": ["Event", "Rate"],
        "rows": [["EventA", "6%"], ["EventB", "4%"]],
        "llm_understanding": {
            "applicability_cues": ["event summary"],
            "applicability_attributes": [{"key": "section", "value": "Events"}],
        },
    }
    decision = select_eligible_table_chunks(
        AXIS_QUERY,
        [_chunk(matrix), _chunk(unitless)],
        get_table=lambda c: c["table"],
    )
    states = {c.table_id: c for c in decision.candidates}
    assert states["t_list"].state == INELIGIBLE
    assert "incompatible_header_units" in states["t_list"].reasons
    assert [c["table_id"] for c in decision.eligible_chunks] == ["t_matrix"]


# 8. OCR-damaged numerical ranges.
def test_ocr_damaged_range_is_parsed_and_matched():
    damaged = parse_range_label(">300 400")
    assert damaged is not None
    assert damaged.low == 300.0 and damaged.high == 400.0
    assert damaged.inclusive_low is False
    assert range_contains(damaged, 350.0)
    assert not range_contains(damaged, 300.0)

    table = {
        "table_id": "t_ocr",
        "page": 1,
        "headers": ["Assay", "Value"],
        "rows": [[">200-300", "150"], [">300 400", "300"]],
    }
    cell = lookup_cell_in_table(table, "What value for assay 350?")
    assert isinstance(cell, VerifiedCell)
    assert cell.value == "300"
    assert cell.row_index == 1


# 9. Merged / hierarchical headers.
def test_hierarchical_header_path_is_preserved_in_evidence():
    table = {
        "table_id": "t_hier",
        "page": 6,
        "headers": ["Assay (IU/mL)", "Schedule", "30-60 kg", ">60-70 kg"],
        "normalized_header_paths": [
            ["Assay"],
            ["Schedule"],
            ["Dose", "30-60 kg"],
            ["Dose", ">60-70 kg"],
        ],
        "rows": [[">300-400", "Every 2 weeks", "225", "300"]],
    }
    cell = lookup_cell_in_table(
        table, "What value for assay 350 IU/mL and body weight 65 kg?"
    )
    assert isinstance(cell, VerifiedCell)
    assert cell.col_index == 3
    assert cell.value == "300"
    assert cell.evidence["normalized_header_paths"]["column"] == [
        "Dose",
        ">60-70 kg",
    ]


# 10. Multiple matching rows with conflicting values.
def test_overlapping_rows_with_conflicting_values_are_ambiguous():
    table = {
        "table_id": "t_dup",
        "page": 2,
        "headers": ["Assay", "Value"],
        "rows": [[">300-400", "150"], [">300-400", "300"]],
    }
    outcome = lookup_cell_in_table(table, "What value for assay 350?")
    assert not isinstance(outcome, VerifiedCell)
    assert outcome.status == STATUS_AMBIGUOUS
    assert outcome.candidate_values == ["150", "300"]
    assert "t_dup" in outcome.candidate_table_ids


# 11. Multiple candidates with identical verified values.
def test_identical_verified_values_are_accepted_with_provenance():
    left = _period_table("t_1", 3, "Lookup grid — FY2024", "TrialOne", "300")
    right = _period_table("t_2", 4, "Lookup grid — FY2024", "TrialOne", "300")
    result = interpret_table_question(AXIS_QUERY, [_chunk(left), _chunk(right)])
    assert result.verified is not None
    assert result.verified.value == "300"
    agreeing = result.sources_update["verified_cell"]["agreeing_table_ids"]
    assert set(agreeing) == {"t_1", "t_2"}


# 12. Malformed table structures.
def test_malformed_structure_is_invalid_and_not_looked_up():
    malformed = {
        "table_id": "t_bad",
        "page": 1,
        "headers": ["column 1", "column 2"],
        "rows": [["a", "1"], ["b"]],
    }
    outcome = lookup_cell_in_table(malformed, "What value for assay 350?")
    assert not isinstance(outcome, VerifiedCell)
    assert outcome.status == STATUS_INVALID_TABLE_STRUCTURE
    assert outcome.structural_validation["status"]


# 13. A definitive answer followed by an ambiguous verification result.
def test_definitive_answer_replaced_by_ambiguous_verification():
    sources = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_AMBIGUOUS,
            "candidate_values": ["300", "450"],
            "candidate_table_ids": ["t_a", "t_b"],
            "missing_dimensions": ["population"],
            "reason": "ambiguous_applicability",
            "verification_status": "ambiguous",
        }
    ]
    answer = (
        "**Clinical Takeaway**\nThe label recommends 300 mg every 4 weeks.\n"
        "**Classification**\n| Off-label | No |\n"
    )
    out = _reconcile_answer_with_verified_cell(answer, sources)
    assert out.startswith("Inconclusive")
    assert "Clinical Takeaway" not in out
    assert "300 mg every 4 weeks" not in out
    assert "population" in out


# 14. Streaming generation with ambiguous evidence.
def test_streaming_gate_blocks_ambiguous_evidence_before_tokens():
    ambiguous = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_AMBIGUOUS,
            "candidate_values": ["300", "450"],
            "candidate_table_ids": ["t_a", "t_b"],
            "missing_dimensions": ["population"],
            "reason": "ambiguous_applicability",
            "verification_status": "ambiguous",
        }
    ]
    msg = _streaming_gate_message(ambiguous)
    assert msg is not None and msg.startswith("Inconclusive")
    assert "450" in msg

    verified = [
        {"verified_cell": {"value": "300", "status": "matched", "page": 4}}
    ]
    assert _streaming_gate_message(verified) is None

    soft = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": True,
            "status": "not_found",
        }
    ]
    assert _streaming_gate_message(soft) is None


# 15. Reconciliation attempting to restore an unsupported conclusion.
def test_reconcile_is_idempotent_and_cannot_restore_conclusion():
    sources = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_AMBIGUOUS,
            "candidate_values": ["225", "450"],
            "candidate_table_ids": ["t_a", "t_b"],
            "missing_dimensions": ["population"],
        }
    ]
    model_output = (
        "Authoritative verified table value: 450. On-label: Yes. "
        "The applicable regimen is 450 mg every 2 weeks."
    )
    first = _reconcile_answer_with_verified_cell(model_output, sources)
    assert first.startswith("Inconclusive")
    assert "450 mg every 2 weeks" not in first
    assert "On-label: Yes" not in first
    # Re-running the gate must not append or re-wrap.
    second = _reconcile_answer_with_verified_cell(first, sources)
    assert second == first


# 16. Valid evidence that must not be rejected for irrelevant metadata.
def test_valid_evidence_survives_irrelevant_metadata():
    table = _sibling("t_valid", 5, "AlphaCohort", "300")
    table["llm_understanding"]["applicability_attributes"].append(
        {"key": "page_layout", "value": "landscape appendix"}
    )
    table["document_version"] = "v3"
    result = interpret_table_question(AXIS_QUERY, [_chunk(table)])
    assert result.verified is not None
    assert result.verified.value == "300"
    record = result.verified.provenance_record()
    assert record["document_version"] == "v3"
    assert record["page"] == 5
    assert record["table_id"] == "t_valid"
    assert record["row_index"] == 1
    assert record["col_index"] == 2
    assert record["original_cell_text"] == "300"
    assert record["normalized_row_headers"] == [">300-400"]
    assert record["verification_status"] == "matched"
    assert record["structural_validation"]["status"]


# ── Global answer state + observability ─────────────────────────────────────


def test_answer_state_precedence_and_mapping():
    verified = [
        {"verified_cell": {"value": "300", "status": "matched", "page": 4}}
    ]
    assert answer_state_from_sources(verified).state == ANSWER_STATE_VERIFIED
    assert answer_state_from_sources([]).state == ANSWER_STATE_NOT_APPLICABLE
    assert (
        answer_state_from_sources(
            [
                {
                    "insufficient_table_evidence": True,
                    "soft_insufficient": True,
                    "status": "not_found",
                }
            ]
        ).state
        == ANSWER_STATE_INSUFFICIENT
    )
    invalid = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_INVALID_TABLE_STRUCTURE,
        }
    ]
    decision = answer_state_from_sources(invalid)
    assert decision.state == ANSWER_STATE_INVALID
    assert decision.blocks_definitive_answer is True
    # A stale verified cell cannot override a hard failure.
    mixed = invalid + verified
    assert answer_state_from_sources(mixed).state == ANSWER_STATE_INVALID


def test_invalid_structure_gate_mentions_insufficient_evidence():
    sources = [
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": STATUS_INVALID_TABLE_STRUCTURE,
            "candidate_table_ids": ["t_bad"],
        }
    ]
    out = _reconcile_answer_with_verified_cell("The answer is 42.", sources)
    assert "Insufficient table evidence" in out
    assert "42" not in out


def test_eligibility_and_verification_observability():
    clear_table_rag_trace()
    begin_table_rag_trace(question=AXIS_QUERY, document_id="doc-1")
    tables = [
        _sibling("t_a", 1, "AlphaCohort", "300"),
        _sibling("t_b", 2, "BetaCohort", "450"),
    ]
    decision = select_eligible_table_chunks(
        AXIS_QUERY, [_chunk(t) for t in tables], get_table=lambda c: c["table"]
    )
    record_candidate_audit(
        retrieved_table_ids=[c["table_id"] for c in decision.eligible_chunks],
        eligibility=[c.to_dict() for c in decision.candidates],
    )
    record_cell_verification(
        table_id="t_a",
        row_index=1,
        col_index=2,
        row_headers=[">300-400"],
        col_headers=[">60-70 kg"],
        status="matched",
    )
    record_answer_state(ANSWER_STATE_AMBIGUOUS)
    trace = current_table_rag_trace()
    assert trace is not None
    assert trace.eligibility_decisions
    assert trace.remaining_conflicts == ["t_a,t_b"]
    assert trace.numeric_match["row_index"] == 1
    assert trace.cell_verification["status"] == "matched"
    assert trace.answer_state == ANSWER_STATE_AMBIGUOUS
    clear_table_rag_trace()


@pytest.mark.parametrize(
    "value,expected",
    [(">300-400", 350.0), ("300-400", 300.0), ("<300", 250.0)],
)
def test_range_contains_boundaries(value, expected):
    parsed = parse_range_label(value)
    assert parsed is not None
    assert range_contains(parsed, expected)
