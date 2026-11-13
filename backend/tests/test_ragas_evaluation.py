"""Unit tests for RAGAS evaluation helpers (mocked LLM/embeddings).

Live RAGAS judge metrics are integration-only — see test_ragas_integration.py marker.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.rag.evaluation.runner import (
    aggregate_table_metrics,
    build_ragas_rows,
    evaluate_dataset,
    merge_overall_report,
    summarize_ragas_result,
)
from app.rag.evaluation.table_metrics import (
    evaluate_table_case,
    numeric_answer_match,
    page_attribution_ok,
    unit_preserved,
)


def test_numeric_and_unit_metrics():
    assert numeric_answer_match("300 mg", "Dose is 300 mg every 4 weeks") is True
    assert numeric_answer_match("6659.1", "Sum = 6,659.10") is True
    assert numeric_answer_match("150", "225 mg") is False
    assert unit_preserved("2 mL", "Total volume is 2 mL") is True
    assert unit_preserved("150", "150 every 4 weeks") is None


def test_page_attribution():
    assert page_attribution_ok(5, [{"page": 5}]) is True
    assert page_attribution_ok(5, [{"verified_cell": {"page": 5}}]) is True
    assert page_attribution_ok(5, [{"page": 2}]) is False
    assert page_attribution_ok(None, [{"page": 1}]) is None


def test_evaluate_table_case_and_aggregate():
    case = {
        "reference": "300 mg",
        "expected_page": 5,
        "expected_table": "tbl_abc",
        "tags": ["single_cell"],
    }
    metrics = evaluate_table_case(
        case,
        answer="The dose is 300 mg every 4 weeks.",
        sources=[
            {
                "page": 5,
                "table_id": "tbl_abc",
                "selector_status": "matched",
                "verified_cell": {"value": "300", "page": 5, "table_id": "tbl_abc"},
            }
        ],
        verified={"value": "300", "page": 5, "table_id": "tbl_abc"},
    )
    assert metrics["numeric_answer_correct"] is True
    assert metrics["page_attribution_ok"] is True
    assert metrics["table_id_selected"] is True

    agg = aggregate_table_metrics(
        [{"id": "a", "table_metrics": metrics, "success": True, "retrieved_contexts": ["x"], "response": "300 mg"}]
    )
    assert agg["numeric_answer_correct"]["accuracy"] == 1.0
    assert agg["numeric_answer_correct"]["measured"] == 1


def test_build_ragas_rows_skips_failures_and_respects_reference():
    rows = build_ragas_rows(
        [
            {
                "success": False,
                "user_input": "q",
                "response": "",
                "retrieved_contexts": [],
                "has_reference": True,
                "reference": "1",
            },
            {
                "success": True,
                "user_input": "q1",
                "response": "a1",
                "retrieved_contexts": ["c1"],
                "has_reference": True,
                "reference": "150",
            },
            {
                "success": True,
                "user_input": "q2",
                "response": "a2",
                "retrieved_contexts": ["c2"],
                "has_reference": False,
                "reference": None,
            },
        ]
    )
    assert len(rows) == 2
    assert "reference" in rows[0]
    assert "reference" not in rows[1]


def test_summarize_ragas_result_from_dict_and_scores_list():
    assert summarize_ragas_result({"faithfulness": 0.5, "x": "skip"}) == {
        "faithfulness": 0.5,
        "x": None,
    }
    assert summarize_ragas_result(
        SimpleNamespace(scores=[{"faithfulness": 1.0}, {"faithfulness": 0.5}])
    ) == {"faithfulness": 0.75}


def test_evaluate_dataset_report_generation(tmp_path, monkeypatch):
    dataset = {
        "name": "unit_fixture",
        "user_id": "user-1",
        "cases": [
            {
                "id": "c1",
                "user_input": "What is the dose?",
                "reference": "150",
                "expected_page": 1,
            }
        ],
    }
    path = tmp_path / "unit_fixture.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    def _fake_case(case, *, user_id, llm_provider=None, top_k=None):
        return {
            "id": case["id"],
            "user_input": case["user_input"],
            "reference": case.get("reference"),
            "has_reference": True,
            "reference_contexts": [],
            "response": "150 mg",
            "retrieved_contexts": ["dose table: 150"],
            "sources": [{"page": 1, "selector_status": "matched"}],
            "chunks_meta": [{"page": 1}],
            "verified_cell": {"value": "150", "page": 1},
            "selector_status": "matched",
            "interpretation": {},
            "document_id": "doc-1",
            "original_name": "x.pdf",
            "expected_page": 1,
            "expected_table": None,
            "tags": [],
            "eval_mode": "reference",
            "latency_ms": 1.0,
            "error": None,
            "success": True,
        }

    monkeypatch.setattr(
        "app.rag.evaluation.runner.run_live_pipeline_case",
        _fake_case,
    )
    monkeypatch.setattr(
        "app.rag.evaluation.runner.run_ragas_metrics",
        lambda *a, **k: {
            "status": "skipped",
            "reason": "unit_test",
            "scores": {},
            "metrics_run": [],
            "metrics_unavailable": ["all"],
        },
    )

    report_dir = tmp_path / "reports"
    report = evaluate_dataset(
        path,
        skip_llm_metrics=True,
        report_dir=report_dir,
    )
    assert report["n_questions"] == 1
    assert report["successful_queries"] == 1
    assert Path(report["report_path"]).exists()
    assert Path(report["summary_path"]).exists()
    written = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert written["notes"]["fabricated_scores"] is False
    assert written["table_specific_metrics"]["numeric_answer_correct"]["accuracy"] == 1.0

    overall = merge_overall_report([report], report_dir=report_dir)
    assert Path(overall["report_path"]).exists()
    assert overall["n_questions"] == 1


def test_ragas_compat_allows_import():
    from app.rag.evaluation.ragas_compat import ensure_ragas_langchain_compat

    ensure_ragas_langchain_compat()
    import ragas

    assert hasattr(ragas, "evaluate") or hasattr(ragas, "__version__")


def test_datasets_exist_and_schema():
    root = Path(__file__).resolve().parents[1] / "data" / "evals" / "ragas" / "datasets"
    for name in ("medical.json", "invoice.json", "research.json", "finance.json"):
        path = root / name
        assert path.exists(), path
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "cases" in data
        for case in data["cases"]:
            assert "user_input" in case
            # reference may be null for retrieval-only
            assert "reference" in case or case.get("eval_mode") == "retrieval_only"
