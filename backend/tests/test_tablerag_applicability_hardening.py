"""Cross-domain regressions for applicability, conflicts, and selector gate."""
from __future__ import annotations

from app.rag.generation.structured_lookup import (
    STATUS_AMBIGUOUS,
    STATUS_EMPTY_CELL,
    STATUS_MATCHED,
    STATUS_VERIFICATION_FAILED,
    ParsedRange,
)
from app.rag.generation.table_calculator import compute_from_operands
from app.rag.generation.table_selection import (
    TableApplicability,
    bound_sets_are_distinct,
    profiles_are_distinct,
    ranges_are_equivalent,
    ranges_overlap,
    select_eligible_table_chunks,
)
from app.rag.generation.table_selector import (
    interpret_with_selector,
    scope_rank_candidates,
    verify_lookup_decision,
)
from app.rag.table_rag.schema import TableLookupDecision


def _chunk(table: dict, score: float = 0.5) -> dict:
    return {
        "table_id": table["table_id"],
        "table": table,
        "caption": table.get("caption") or "",
        "page": table.get("page"),
        "score": score,
        "chunk_type": "table",
        "document_id": "doc-test",
    }


def _group_tables():
    """Identical numeric axes; different population attributes.

    Population labels use multi-character tokens so ``_tokens`` retains them
    (single letters are filtered by the tokenizer).
    """
    base_headers = ["Assay (IU/mL)", "70-90 kg"]
    tables = []
    for tid, pop, page, val in (
        ("t1", "AlphaCohort", 1, "100"),
        ("t2", "BetaCohort", 2, "200"),
        ("t3", "GammaCohort", 3, "300"),
    ):
        tables.append(
            {
                "table_id": tid,
                "page": page,
                "caption": f"Rates for {pop}",
                "headers": base_headers,
                "rows": [[">200-300", val], [">300-400", "400"]],
                "llm_understanding": {
                    "purpose": f"Rates for {pop}",
                    "applicability_cues": [pop],
                    "applicability_attributes": [
                        {"key": "population", "value": pop}
                    ],
                    "differentiating_dimensions": ["population"],
                },
            }
        )
    return tables


def test_identical_axes_missing_population_is_ambiguous():
    tables = _group_tables()
    chunks = [_chunk(t) for t in tables]
    # Avoid document_id so sibling SQL expand is skipped in unit tests.
    for c in chunks:
        c.pop("document_id", None)
    decision = select_eligible_table_chunks(
        "What is the value for 78 kg and 250 IU/mL?",
        chunks,
        get_table=lambda c: c["table"],
    )
    assert decision.requires_disambiguation
    assert "population" in " ".join(decision.missing_dimensions).lower() or decision.missing_dimensions

    outcome, sel, _ = interpret_with_selector(
        "What is the value for 78 kg and 250 IU/mL from the table?",
        chunks,
    )
    assert sel.status == "needs_disambiguation"
    assert getattr(outcome, "status", None) == STATUS_AMBIGUOUS
    assert set(outcome.candidate_table_ids) >= {"t1", "t2", "t3"}
    # Must surface concrete values — not silently pick one table.
    assert outcome.candidate_values
    assert len(set(outcome.candidate_values)) >= 2


def test_explicit_population_selects_unique_table():
    tables = _group_tables()
    chunks = [_chunk(t, score=0.9 if t["table_id"] != "t2" else 0.1) for t in tables]
    for c in chunks:
        c.pop("document_id", None)
    outcome, sel, cands = interpret_with_selector(
        "What is the value for BetaCohort at 78 kg and 250 IU/mL from the table?",
        chunks,
    )
    assert sel.status == "matched"
    assert getattr(outcome, "status", STATUS_MATCHED) == STATUS_MATCHED
    assert outcome.table_id == "t2"
    assert str(outcome.value) == "200"


def test_overlapping_bounds_are_distinct_classes():
    a = ParsedRange(low=0, high=2, unit="years", raw="0-2 years")
    b = ParsedRange(low=0, high=5, unit="years", raw="0-5 years")
    assert ranges_overlap(a, b)
    assert not ranges_are_equivalent(a, b)
    assert bound_sets_are_distinct([a], [b])

    left = TableApplicability(
        table_id="a",
        bound_ranges=[a],
        cue_tokens={"cohort", "a"},
        attributes={"cohort": "0-2"},
    )
    right = TableApplicability(
        table_id="b",
        bound_ranges=[b],
        cue_tokens={"cohort", "b"},
        attributes={"cohort": "0-5"},
    )
    assert profiles_are_distinct(left, right)


def test_non_overlapping_and_equivalent_bounds():
    a = ParsedRange(low=0, high=2, unit="years", inclusive_high=False, raw="<2 years")
    b = ParsedRange(low=2, high=5, unit="years", inclusive_low=True, raw=">=2-5 years")
    assert not ranges_overlap(a, b)
    assert bound_sets_are_distinct([a], [b])

    same = ParsedRange(low=0, high=2, unit="years", raw="0-2 years")
    assert ranges_are_equivalent(a.__class__(0, 2, "years"), same) or ranges_are_equivalent(
        ParsedRange(0, 2, "years"), same
    )
    assert not bound_sets_are_distinct(
        [ParsedRange(0, 2, "years")], [ParsedRange(0, 2, "years")]
    )


def test_unit_incompatibility_makes_bounds_distinct():
    a = ParsedRange(low=0, high=10, unit="years")
    b = ParsedRange(low=0, high=10, unit="kg")
    assert bound_sets_are_distinct([a], [b])


def test_adult_content_token_does_not_beat_indication():
    """Soft rank: content token 'asthma' beats cohort-only 'adult' sibling."""
    asthma = {
        "table_id": "t_asthma",
        "page": 5,
        "caption": "Table 1. Doses for Patients 12 Years and Older with Asthma",
        "headers": ["Assay (IU/mL)", "70-90 kg"],
        "rows": [[">200-300", "225"]],
        "llm_understanding": {
            "purpose": "asthma dosing",
            "applicability_cues": ["asthma", "12 years and older"],
            "applicability_attributes": [{"key": "indication", "value": "asthma"}],
        },
    }
    other = {
        "table_id": "t_other",
        "page": 7,
        "caption": "Table 3. Doses for Adult Patients with CRSwNP",
        "headers": ["Assay (IU/mL)", "70-90 kg"],
        "rows": [[">200-300", "450"]],
        "llm_understanding": {
            "purpose": "CRSwNP dosing",
            "applicability_cues": ["CRSwNP", "adult"],
            "applicability_attributes": [{"key": "indication", "value": "CRSwNP"}],
        },
    }
    raw = [
        {
            "table_id": "t_other",
            "table": other,
            "chunk": _chunk(other, 0.9),
            "score": 0.9,
        },
        {
            "table_id": "t_asthma",
            "table": asthma,
            "chunk": _chunk(asthma, 0.5),
            "score": 0.5,
        },
    ]
    ranked = scope_rank_candidates(
        "Adult asthma, 80 kg, IgE 250 IU/mL from the dosing table",
        raw,
    )
    assert ranked[0]["table_id"] == "t_asthma"


def test_wrong_table_coords_rejected_when_not_eligible():
    """Coordinates on an excluded sibling must not verify as matched."""
    tables = _group_tables()
    # Only Group B is eligible for this question; force a decision on Group A.
    chunks = [_chunk(t) for t in tables]
    decision = TableLookupDecision(
        status="matched",
        table_id="t1",
        row_index=0,
        col_index=1,
        rationale="forced wrong table",
    )
    # Build candidate list as selector would for eligible-only Group B.
    eligible = [
        {
            "table_id": "t2",
            "table": tables[1],
            "chunk": _chunk(tables[1]),
            "score": 0.5,
        }
    ]
    out = verify_lookup_decision(
        decision,
        eligible,
        question="Group B at 78 kg and 250 IU/mL",
    )
    assert getattr(out, "status", None) == STATUS_VERIFICATION_FAILED


def test_empty_cell_status():
    table = {
        "table_id": "empty",
        "page": 1,
        "caption": "Sparse grid",
        "headers": ["X", "Y"],
        "rows": [["10", ""], ["20", "N/A"]],
    }
    decision = TableLookupDecision(
        status="matched",
        table_id="empty",
        row_index=0,
        col_index=1,
        rationale="empty",
    )
    out = verify_lookup_decision(
        decision,
        [{"table_id": "empty", "table": table, "chunk": _chunk(table)}],
    )
    assert out.status == STATUS_EMPTY_CELL


def test_calculator_rejects_mixed_units():
    table = {
        "table_id": "mix",
        "headers": ["A", "B"],
        "rows": [["1", "10 mg"], ["2", "20 kg"]],
    }
    try:
        compute_from_operands(table, [(0, 1), (1, 1)], operation="sum")
        assert False, "expected unit mismatch"
    except ValueError as exc:
        assert "incompatible operand units" in str(exc)


def test_complementary_same_class_not_forced_ambiguous():
    """Same applicability attributes → complementary, not a conflict gate."""
    a = {
        "table_id": "dup1",
        "page": 1,
        "caption": "Revenue by region — FY2024",
        "headers": ["Region", "Amount (USD)"],
        "rows": [["APAC", "10"], ["EMEA", "12"]],
        "llm_understanding": {
            "applicability_cues": ["FY2024", "revenue"],
            "applicability_attributes": [
                {"key": "reporting_period", "value": "FY2024"}
            ],
            "differentiating_dimensions": ["reporting_period"],
        },
    }
    b = {
        "table_id": "dup2",
        "page": 2,
        "caption": "Revenue by region — FY2024 (continued)",
        "headers": ["Region", "Amount (USD)"],
        "rows": [["AMER", "15"], ["LATAM", "8"]],
        "llm_understanding": {
            "applicability_cues": ["FY2024", "revenue"],
            "applicability_attributes": [
                {"key": "reporting_period", "value": "FY2024"}
            ],
            "differentiating_dimensions": ["reporting_period"],
        },
    }
    left = TableApplicability(
        table_id="dup1",
        attributes={"reporting_period": "FY2024"},
        cue_tokens={"fy2024", "revenue", "apac"},
        purpose_tokens={"revenue", "region", "fy2024"},
    )
    right = TableApplicability(
        table_id="dup2",
        attributes={"reporting_period": "FY2024"},
        cue_tokens={"fy2024", "revenue", "amer"},
        purpose_tokens={"revenue", "region", "fy2024"},
    )
    # Identical reporting_period attribute → not distinct classes.
    assert not profiles_are_distinct(left, right)
