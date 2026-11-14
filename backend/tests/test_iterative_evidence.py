"""Unit tests for iterative evidence retrieval orchestration."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from app.rag.generation import iterative_evidence as ie
from app.rag.generation.llm_table_reasoning import TableInterpretationResult
from app.rag.generation.table_evidence import (
    STATUS_MATCHED,
    STATUS_NOT_FOUND,
    InsufficientEvidence,
    VerifiedCell,
)


def test_union_chunks_dedupes_and_prefers_fuller_grid():
    a = {
        "id": "c1",
        "table_id": "t1",
        "score": 0.1,
        "table": {"rows": [["1"]]},
    }
    b = {
        "id": "c1",
        "table_id": "t1",
        "score": 0.2,
        "table": {"rows": [["1"], ["2"], ["3"]]},
    }
    c = {"id": "c2", "table_id": "t2", "table": {"rows": [["x"]]}}
    merged = ie.union_chunks([a], [b, c])
    assert len(merged) == 2
    assert merged[0]["id"] == "c1"
    assert len(merged[0]["table"]["rows"]) == 3
    assert merged[1]["id"] == "c2"


def test_followup_queries_prefer_llm_additional_search_queries():
    result = TableInterpretationResult(
        is_structured_lookup=True,
        result=InsufficientEvidence(
            reason="need more",
            status=STATUS_NOT_FOUND,
            missing_dimensions=["body weight"],
        ),
        raw_llm={
            "additional_search_queries": [
                "Table 1 asthma dosing IgE weight",
                "footnote every 2 or 4 weeks",
            ],
            "missing_information": ["body weight"],
        },
    )
    qs = ie._followup_queries("what dose?", result)
    assert qs[0].startswith("Table 1")
    assert any("footnote" in q.lower() for q in qs)


def test_should_iterate_false_when_verified():
    result = TableInterpretationResult(
        is_structured_lookup=True,
        result=VerifiedCell(
            value="225",
            row_headers=["a"],
            col_headers=["b"],
            row_index=0,
            col_index=1,
            status=STATUS_MATCHED,
        ),
    )
    assert ie._should_iterate(result) is False


def test_run_iterative_loop_fetches_once_then_verifies(monkeypatch):
    calls = {"interpret": 0, "retrieve": 0}

    insuff = TableInterpretationResult(
        is_structured_lookup=True,
        result=InsufficientEvidence(
            reason="missing axis",
            status=STATUS_NOT_FOUND,
            missing_dimensions=["weight band"],
        ),
        raw_llm={
            "status": "insufficient_evidence",
            "missing_information": ["weight band"],
            "additional_search_queries": ["weight band dosing table"],
            "answer_ready": False,
        },
    )
    verified = TableInterpretationResult(
        is_structured_lookup=True,
        result=VerifiedCell(
            value="225",
            row_headers=[">200-300"],
            col_headers=[">60-70 kg"],
            row_index=2,
            col_index=3,
            status=STATUS_MATCHED,
            table_id="t1",
        ),
        raw_llm={"status": "answered", "answer_ready": True},
    )

    def fake_interpret(question, chunks):
        calls["interpret"] += 1
        return insuff if calls["interpret"] == 1 else verified

    def fake_retrieve(**kwargs):
        calls["retrieve"] += 1
        return [
            {
                "id": "new1",
                "document_id": "d1",
                "table_id": "t1",
                "page": 5,
                "text": "dosing table",
                "table": {
                    "table_id": "t1",
                    "headers": ["IgE", "Dose"],
                    "rows": [[">200-300", "225"]],
                },
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(ie, "interpret_table_question", fake_interpret)
    monkeypatch.setattr(
        ie,
        "get_settings",
        lambda: SimpleNamespace(
            TABLE_RAG_ITERATIVE_ENABLED=True,
            TABLE_RAG_MAX_ITERATIONS=2,
            TABLE_RAG_ITERATIVE_TOP_K=4,
        ),
    )
    monkeypatch.setattr(
        "app.rag.retrieval.retriever.retrieve",
        fake_retrieve,
    )

    result, merged, stats = ie.run_iterative_evidence_loop(
        "dose for IgE 300 weight 70?",
        [{"id": "seed", "text": "seed"}],
        user_id="u1",
        document_id="d1",
        max_iterations=2,
    )
    assert calls["interpret"] == 2
    assert calls["retrieve"] >= 1
    assert result.verified is not None
    assert result.verified.value == "225"
    assert stats["iterations"] >= 1
    assert any(c.get("id") == "new1" for c in merged)


def test_iterative_disabled_is_single_shot(monkeypatch):
    calls = {"interpret": 0}

    def fake_interpret(question, chunks):
        calls["interpret"] += 1
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=InsufficientEvidence(
                reason="missing",
                status=STATUS_NOT_FOUND,
                missing_dimensions=["x"],
            ),
            raw_llm={"additional_search_queries": ["x"], "answer_ready": False},
        )

    monkeypatch.setattr(ie, "interpret_table_question", fake_interpret)
    monkeypatch.setattr(
        ie,
        "get_settings",
        lambda: SimpleNamespace(
            TABLE_RAG_ITERATIVE_ENABLED=False,
            TABLE_RAG_MAX_ITERATIONS=2,
            TABLE_RAG_ITERATIVE_TOP_K=4,
        ),
    )
    result, merged, stats = ie.run_iterative_evidence_loop(
        "q",
        [{"id": "a"}],
        user_id="u",
        document_id="d",
    )
    assert calls["interpret"] == 1
    assert stats["iterations"] == 0
    assert result.insufficient is not None


def test_apply_context_budget_caps_chunks_and_chars():
    chunks = [
        {"id": f"c{i}", "text": "x" * 100, "score": 0.1 * i}
        for i in range(20)
    ]
    trimmed = ie.apply_context_budget(chunks, max_chunks=5, max_chars=250)
    assert len(trimmed) <= 5
    assert sum(len(c["text"]) for c in trimmed) <= 250 + 100  # first chunk always kept


def test_should_iterate_assessment_adaptive_stop():
    enough = ie.EvidenceAssessment(
        sufficient=True,
        can_answer_safely=True,
        additional_queries=["more"],
        missing_info=["x"],
    )
    assert ie._should_iterate_assessment(enough) is False

    gaps = ie.EvidenceAssessment(
        sufficient=False,
        can_answer_safely=False,
        additional_queries=["find results table"],
        missing_info=["effect size"],
    )
    assert ie._should_iterate_assessment(gaps) is True


def test_agentic_loop_merges_then_stops_when_sufficient(monkeypatch):
    calls = {"interpret": 0, "retrieve": 0, "assess": 0}

    def fake_interpret(question, chunks):
        calls["interpret"] += 1
        return TableInterpretationResult(is_structured_lookup=False)

    assessments = [
        ie.EvidenceAssessment(
            sufficient=False,
            can_answer_safely=False,
            missing_info=["key finding"],
            additional_queries=["key quantitative finding table"],
            rationale="gaps",
        ),
        ie.EvidenceAssessment(
            sufficient=True,
            can_answer_safely=True,
            evidence_ids=["new1"],
            doc_ids=["d1"],
            pages=[2],
            text_refs=["research.pdf"],
            table_refs=[],
            missing_info=[],
            additional_queries=[],
            rationale="enough",
        ),
    ]

    def fake_assess(question, chunks, **kwargs):
        calls["assess"] += 1
        idx = min(calls["assess"] - 1, len(assessments) - 1)
        return assessments[idx]

    def fake_retrieve(**kwargs):
        calls["retrieve"] += 1
        return [
            {
                "id": "new1",
                "document_id": "d1",
                "page": 2,
                "filename": "research.pdf",
                "text": "The key quantitative finding is 12.5% improvement.",
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(ie, "interpret_table_question", fake_interpret)
    monkeypatch.setattr(ie, "assess_evidence_sufficiency", fake_assess)
    monkeypatch.setattr(
        ie,
        "get_settings",
        lambda: SimpleNamespace(
            TABLE_RAG_ITERATIVE_ENABLED=True,
            TABLE_RAG_MAX_ITERATIONS=2,
            TABLE_RAG_ITERATIVE_TOP_K=4,
            TOP_K_RERANK=8,
            TABLE_RAG_CONTEXT_CHAR_BUDGET=24000,
        ),
    )
    monkeypatch.setattr("app.rag.retrieval.retriever.retrieve", fake_retrieve)

    result, merged, stats = ie.run_iterative_evidence_loop(
        "Summarize a key quantitative finding",
        [{"id": "seed", "text": "intro only", "document_id": "d1"}],
        user_id="u1",
        document_id="d1",
        max_iterations=2,
    )
    assert result.is_structured_lookup is False
    assert calls["assess"] >= 2
    assert calls["retrieve"] >= 1
    assert stats["agentic_iterations"] >= 1
    assert stats["evidence_assessment"]["sufficient"] is True
    assert any(c.get("id") == "new1" for c in merged)
    # Adaptive stop: should not keep retrieving after sufficient.
    assert calls["retrieve"] == 1


def test_agentic_skipped_when_table_verified(monkeypatch):
    calls = {"assess": 0}

    def fake_interpret(question, chunks):
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=VerifiedCell(
                value="225",
                row_headers=["a"],
                col_headers=["b"],
                row_index=0,
                col_index=1,
                status=STATUS_MATCHED,
            ),
            raw_llm={"status": "answered", "answer_ready": True},
        )

    def fake_assess(*_a, **_k):
        calls["assess"] += 1
        return ie.EvidenceAssessment(sufficient=False, additional_queries=["x"])

    monkeypatch.setattr(ie, "interpret_table_question", fake_interpret)
    monkeypatch.setattr(ie, "assess_evidence_sufficiency", fake_assess)
    monkeypatch.setattr(
        ie,
        "get_settings",
        lambda: SimpleNamespace(
            TABLE_RAG_ITERATIVE_ENABLED=True,
            TABLE_RAG_MAX_ITERATIONS=2,
            TABLE_RAG_ITERATIVE_TOP_K=4,
            TOP_K_RERANK=8,
        ),
    )
    result, _merged, stats = ie.run_iterative_evidence_loop(
        "dose?",
        [{"id": "seed", "text": "table"}],
        user_id="u",
        document_id="d",
    )
    assert result.verified is not None
    assert calls["assess"] == 0
    assert stats["agentic_iterations"] == 0


def test_soft_insufficient_triggers_agentic_assessment(monkeypatch):
    calls = {"assess": 0, "retrieve": 0}

    def fake_interpret(question, chunks):
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=InsufficientEvidence(
                reason="No structured tables in retrieved chunks",
                status=STATUS_NOT_FOUND,
                soft=True,
            ),
            raw_llm={"soft_insufficient": True, "answer_ready": False},
        )

    def fake_assess(question, chunks, **kwargs):
        calls["assess"] += 1
        return ie.EvidenceAssessment(
            sufficient=True,
            can_answer_safely=True,
            evidence_ids=["seed"],
            rationale="prose enough",
        )

    monkeypatch.setattr(ie, "interpret_table_question", fake_interpret)
    monkeypatch.setattr(ie, "assess_evidence_sufficiency", fake_assess)
    monkeypatch.setattr(
        ie,
        "get_settings",
        lambda: SimpleNamespace(
            TABLE_RAG_ITERATIVE_ENABLED=True,
            TABLE_RAG_MAX_ITERATIONS=2,
            TABLE_RAG_ITERATIVE_TOP_K=4,
            TOP_K_RERANK=8,
        ),
    )

    def fake_retrieve(**kwargs):
        calls["retrieve"] += 1
        return []

    monkeypatch.setattr("app.rag.retrieval.retriever.retrieve", fake_retrieve)

    _result, _merged, stats = ie.run_iterative_evidence_loop(
        "what does the prose say?",
        [{"id": "seed", "text": "prose answer is 42 units"}],
        user_id="u",
        document_id="d",
    )
    assert calls["assess"] >= 1
    assert stats["evidence_assessment"]["sufficient"] is True
    # Sufficient on first assessment → no retrieve.
    assert calls["retrieve"] == 0
