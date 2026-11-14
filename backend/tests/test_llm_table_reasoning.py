"""Tests for LLM-driven table reasoning + Python evidence verification."""
from __future__ import annotations

from app.rag.generation.llm_table_reasoning import (
    collect_table_candidates,
    interpret_table_question,
    is_structured_lookup_question,
    verify_llm_evidence,
)
from app.rag.generation.table_evidence import (
    ANSWER_STATE_AMBIGUOUS,
    ANSWER_STATE_VERIFIED,
    STATUS_MATCHED,
    STATUS_VERIFICATION_FAILED,
    VerifiedCell,
    answer_state_from_sources,
)


def _finance_chunk() -> dict:
    return {
        "chunk_id": "c_fin_1",
        "filename": "finance.pdf",
        "page": 2,
        "chunk_type": "table",
        "table_id": "t_fin_1",
        "caption": "FY2024 Segment Revenue (USD millions)",
        "score": 0.9,
        "table": {
            "table_id": "t_fin_1",
            "caption": "FY2024 Segment Revenue (USD millions)",
            "page": 2,
            "headers": ["Segment", "Q1", "Q2", "Q3", "Q4"],
            "rows": [
                ["Cloud", "120", "135", "150", "160"],
                ["Hardware", "80", "75", "70", "68"],
            ],
        },
        "text": "Table card: FY2024 Segment Revenue",
    }


def _engineering_chunk() -> dict:
    return {
        "chunk_id": "c_eng_1",
        "filename": "materials.pdf",
        "page": 4,
        "chunk_type": "table",
        "table_id": "t_eng_1",
        "caption": "ASTM A36 Tensile Properties",
        "score": 0.8,
        "table": {
            "table_id": "t_eng_1",
            "caption": "ASTM A36 Tensile Properties",
            "page": 4,
            "headers": ["Property", "Value", "Unit"],
            "rows": [
                ["Yield strength", "250", "MPa"],
                ["Tensile strength", "400", "MPa"],
            ],
        },
        "text": "Engineering measurement table",
    }


def test_is_structured_lookup_question_domain_agnostic():
    assert is_structured_lookup_question("What is the Q2 cloud revenue in the table?")
    assert is_structured_lookup_question("Rate for 12 MPa at 40 mm thickness")
    assert not is_structured_lookup_question("Summarize the introduction")


def test_collect_table_candidates_dedupes_and_prefers_rows():
    thin = dict(_finance_chunk())
    thin["table"] = {
        "table_id": "t_fin_1",
        "headers": ["Segment"],
        "rows": [["Cloud"]],
    }
    cands = collect_table_candidates([thin, _finance_chunk(), _engineering_chunk()], top_k=5)
    assert len(cands) == 2
    fin = next(c for c in cands if c["table_id"] == "t_fin_1")
    assert len(fin["table"]["rows"]) == 2


def test_verify_llm_evidence_accepts_real_cell():
    cands = collect_table_candidates([_finance_chunk()])
    payload = {
        "status": "answered",
        "reasoning_summary": "FY2024 Cloud Q2 cell",
        "selected_evidence": [
            {
                "chunk_id": "c_fin_1",
                "table_id": "t_fin_1",
                "row_index": 0,
                "col_index": 2,
                "row_reference": "Cloud",
                "column_reference": "Q2",
                "value": "135",
                "unit": "USD millions",
            }
        ],
        "missing_information": [],
        "answer": "135",
    }
    out = verify_llm_evidence(payload, cands)
    assert isinstance(out, VerifiedCell)
    assert out.value == "135"
    assert out.table_id == "t_fin_1"
    assert out.status == STATUS_MATCHED


def test_verify_prefers_answer_aligned_value_over_schedule():
    """When LLM returns schedule then quantity, pin the quantity cell."""
    chunk = {
        "chunk_id": "c_dose",
        "filename": "label.pdf",
        "page": 5,
        "chunk_type": "table",
        "table_id": "t_dose",
        "caption": "Table 1. Doses by weight",
        "score": 0.9,
        "table": {
            "table_id": "t_dose",
            "caption": "Table 1. Doses by weight",
            "page": 5,
            "headers": ["Band", "Freq", "Dose (mg)"],
            "rows": [[">300-400", "Every 2 weeks", "300"]],
        },
        "text": "dosing table",
    }
    cands = collect_table_candidates([chunk])
    payload = {
        "status": "answered",
        "answer": "300 mg every 2 weeks",
        "selected_evidence": [
            {
                "table_id": "t_dose",
                "chunk_id": "c_dose",
                "row_index": 0,
                "col_index": 1,
                "column_reference": "Freq",
                "value": "Every 2 weeks",
            },
            {
                "table_id": "t_dose",
                "chunk_id": "c_dose",
                "row_index": 0,
                "col_index": 2,
                "column_reference": "Dose (mg)",
                "value": "300",
            },
        ],
    }
    out = verify_llm_evidence(payload, cands)
    assert isinstance(out, VerifiedCell)
    assert out.value == "300"


def test_verify_keeps_insufficient_data_cell_as_answered():
    """Grid cells that say insufficient/blank are valid verified answers."""
    chunk = {
        "chunk_id": "c_blank",
        "filename": "spec.pdf",
        "page": 3,
        "chunk_type": "table",
        "table_id": "t_blank",
        "caption": "Table 1. Limits",
        "score": 0.9,
        "table": {
            "table_id": "t_blank",
            "page": 3,
            "headers": ["Class", "Limit"],
            "rows": [["X", "Insufficient Data to Recommend a Dose"]],
        },
        "text": "limits",
    }
    cands = collect_table_candidates([chunk])
    payload = {
        "status": "insufficient_evidence",
        "answer": "Insufficient Data to Recommend a Dose",
        "reasoning_summary": "cell says insufficient",
        "selected_evidence": [
            {
                "table_id": "t_blank",
                "chunk_id": "c_blank",
                "row_index": 0,
                "col_index": 1,
                "column_reference": "Limit",
                "value": "Insufficient Data to Recommend a Dose",
            }
        ],
    }
    out = verify_llm_evidence(payload, cands)
    assert isinstance(out, VerifiedCell)
    assert "insufficient" in out.value.lower()


def test_verify_llm_evidence_rejects_hallucinated_value():
    cands = collect_table_candidates([_finance_chunk()])
    payload = {
        "status": "answered",
        "reasoning_summary": "invented",
        "selected_evidence": [
            {
                "chunk_id": "c_fin_1",
                "table_id": "t_fin_1",
                "row_index": 0,
                "col_index": 2,
                "value": "9999",
            }
        ],
        "answer": "9999",
    }
    out = verify_llm_evidence(payload, cands)
    assert out.status == STATUS_VERIFICATION_FAILED


def test_verify_rejects_range_header_that_excludes_question_axis():
    """Fail closed when selected band endpoints exclude a unit-bearing factor."""
    chunk = {
        "chunk_id": "c_range",
        "filename": "spec.pdf",
        "page": 5,
        "chunk_type": "table",
        "table_id": "t_range",
        "caption": "Table 1. Lookup by mass and grade",
        "score": 0.9,
        "table": {
            "table_id": "t_range",
            "caption": "Table 1. Lookup by mass and grade",
            "page": 5,
            "headers": ["Grade", ">60-70 kg", ">70-90 kg"],
            "rows": [
                [">200-300", "225", "225"],
                [">300-400", "225", "300"],
            ],
        },
        "text": "lookup",
    }
    cands = collect_table_candidates([chunk])
    # Wrong weight band for 78 kg (">60-70 kg" excludes 78).
    payload = {
        "status": "answered",
        "answer": "225",
        "question": "Value for grade 350 IU/mL at 78 kg?",
        "selected_evidence": [
            {
                "table_id": "t_range",
                "chunk_id": "c_range",
                "row_index": 1,
                "col_index": 1,
                "row_reference": ">300-400",
                "column_reference": ">60-70 kg",
                "value": "225",
            }
        ],
    }
    out = verify_llm_evidence(payload, cands, question=payload["question"])
    assert out.status == STATUS_VERIFICATION_FAILED

    # Correct band for 78 kg.
    payload["selected_evidence"][0]["col_index"] = 2
    payload["selected_evidence"][0]["column_reference"] = ">70-90 kg"
    payload["selected_evidence"][0]["value"] = "300"
    payload["answer"] = "300"
    out_ok = verify_llm_evidence(payload, cands, question=payload["question"])
    assert isinstance(out_ok, VerifiedCell)
    assert out_ok.value == "300"


def test_verify_rejects_exclusive_lower_bound_miss():
    """Exclusive lower bounds like '>300-400' must not accept the endpoint 300."""
    from app.rag.generation.llm_table_reasoning import (
        _axis_hard_conflict,
        _range_covers,
    )

    assert _range_covers(">300-400", 300.0) is False
    assert _range_covers(">200-300", 300.0) is True
    assert _range_covers(">70-90 kg", 78.0) is True
    assert _range_covers(">60-70 kg", 78.0) is False
    conflict = _axis_hard_conflict(
        "IgE 300 IU/mL weight 70 kg",
        row_headers=[">300-400"],
        col_headers=[">60-70 kg"],
    )
    assert conflict
    assert "300" in conflict

    # Prescribed comparator amounts must not poison a covering axis band.
    ok = _axis_hard_conflict(
        "Adult 80 kg, IgE 250 IU/mL prescribed 75 mg",
        row_headers=[">200-300"],
        col_headers=[">70-90 kg"],
    )
    assert ok is None


def test_verify_llm_evidence_ambiguous():
    cands = collect_table_candidates([_finance_chunk(), _engineering_chunk()])
    payload = {
        "status": "ambiguous",
        "reasoning_summary": "Need reporting period",
        "selected_evidence": [
            {"table_id": "t_fin_1", "value": "135"},
            {"table_id": "t_eng_1", "value": "250"},
        ],
        "missing_information": ["reporting_period"],
    }
    out = verify_llm_evidence(payload, cands)
    assert out.status == "ambiguous"
    assert "reporting_period" in out.missing_dimensions


def test_answer_state_from_verified_source():
    sources = [{"verified_cell": {"value": "135", "status": "matched"}}]
    decision = answer_state_from_sources(sources)
    assert decision.state == ANSWER_STATE_VERIFIED


def test_answer_state_hard_ambiguous_outranks_verified():
    sources = [
        {"verified_cell": {"value": "135", "status": "matched"}},
        {
            "insufficient_table_evidence": True,
            "soft_insufficient": False,
            "status": "ambiguous",
            "candidate_values": ["135", "250"],
        },
    ]
    decision = answer_state_from_sources(sources)
    assert decision.state == ANSWER_STATE_AMBIGUOUS


def test_interpret_with_mocked_llm_finance():
    class _Msg:
        content = """{
          "status": "answered",
          "reasoning_summary": "Cloud Q2 revenue cell",
          "selected_chunks": [{"chunk_id": "c_fin_1", "table_id": "t_fin_1", "caption": "FY2024", "relevance": "segment revenue"}],
          "selected_evidence": [{
            "chunk_id": "c_fin_1",
            "table_id": "t_fin_1",
            "row_index": 0,
            "col_index": 2,
            "row_reference": "Cloud",
            "column_reference": "Q2",
            "value": "135",
            "unit": "USD millions"
          }],
          "missing_information": [],
          "answer": "135",
          "citations": []
        }"""

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Client:
        def chat_completion(self, **kwargs):
            return _Resp()

    result = interpret_table_question(
        "What was Cloud segment revenue in Q2 FY2024?",
        [_finance_chunk()],
        client=_Client(),
    )
    assert result.is_structured_lookup
    assert result.verified is not None
    assert result.verified.value == "135"
    assert "verified_cell" in result.sources_update


def test_interpret_engineering_table_with_mocked_llm():
    class _Msg:
        content = """{
          "status": "answered",
          "reasoning_summary": "A36 yield strength",
          "selected_chunks": [{"chunk_id": "c_eng_1", "table_id": "t_eng_1", "caption": "ASTM A36", "relevance": "tensile props"}],
          "selected_evidence": [{
            "chunk_id": "c_eng_1",
            "table_id": "t_eng_1",
            "row_index": 0,
            "col_index": 1,
            "row_reference": "Yield strength",
            "column_reference": "Value",
            "value": "250",
            "unit": "MPa"
          }],
          "missing_information": [],
          "answer": "250 MPa",
          "citations": []
        }"""

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Client:
        def chat_completion(self, **kwargs):
            return _Resp()

    result = interpret_table_question(
        "What is the yield strength for ASTM A36?",
        [_engineering_chunk()],
        client=_Client(),
    )
    assert result.verified is not None
    assert result.verified.value == "250"
    assert result.verified.page == 4
