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
