"""RAGAS + table-metric evaluation runner against the live Genova RAG pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from app.rag.evaluation.pipeline_adapter import load_dataset, run_live_pipeline_case
from app.rag.evaluation.table_metrics import evaluate_table_case


def _safe_mean(values: Sequence[Any]) -> Optional[float]:
    nums = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and v == v  # not NaN
    ]
    if not nums:
        return None
    return round(mean(nums), 4)


def _bool_rate(values: Sequence[Any]) -> Optional[float]:
    booleans = [v for v in values if isinstance(v, bool)]
    if not booleans:
        return None
    return round(sum(1 for v in booleans if v) / len(booleans), 4)


def summarize_ragas_result(result: Any) -> Dict[str, Optional[float]]:
    """Normalize RAGAS result objects into mean metric scores (null if unavailable)."""
    scores: Dict[str, Optional[float]] = {}
    if hasattr(result, "to_pandas"):
        dataframe = result.to_pandas()
        skip = {
            "user_input",
            "retrieved_contexts",
            "response",
            "reference",
            "reference_contexts",
        }
        for column in dataframe.columns:
            if column in skip:
                continue
            scores[str(column)] = _safe_mean(dataframe[column].tolist())
        return scores

    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, (int, float)):
                scores[str(key)] = round(float(value), 4)
            else:
                scores[str(key)] = None
        return scores

    raw_scores = getattr(result, "scores", None)
    if isinstance(raw_scores, list):
        by_metric: Dict[str, List[float]] = {}
        for row in raw_scores:
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                if isinstance(value, (int, float)) and value == value:
                    by_metric.setdefault(str(key), []).append(float(value))
        return {key: _safe_mean(vals) for key, vals in by_metric.items()}

    raise TypeError(f"Unsupported RAGAS result type: {type(result)!r}")


def build_ragas_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert pipeline records into RAGAS EvaluationDataset rows."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        if not rec.get("success"):
            continue
        row: Dict[str, Any] = {
            "user_input": rec.get("user_input") or "",
            "retrieved_contexts": list(rec.get("retrieved_contexts") or []),
            "response": rec.get("response") or "",
        }
        if rec.get("has_reference") and rec.get("reference") is not None:
            row["reference"] = rec["reference"]
        ref_ctx = rec.get("reference_contexts") or []
        if ref_ctx:
            row["reference_contexts"] = list(ref_ctx)
        rows.append(row)
    return rows


def run_ragas_metrics(
    records: Sequence[Dict[str, Any]],
    *,
    llm_provider: Optional[str] = None,
    skip_llm_metrics: bool = False,
) -> Dict[str, Any]:
    """Execute compatible RAGAS metrics. Missing GT → metric omitted (not fabricated)."""
    from app.rag.evaluation.ragas_compat import ensure_ragas_langchain_compat

    ensure_ragas_langchain_compat()

    if skip_llm_metrics:
        return {
            "status": "skipped",
            "reason": "skip_llm_metrics=true",
            "scores": {},
            "metrics_run": [],
            "metrics_unavailable": [
                "context_precision",
                "context_recall",
                "nv_context_relevance",
                "faithfulness",
                "answer_relevancy",
                "answer_correctness",
            ],
        }

    rows = build_ragas_rows(records)
    if not rows:
        return {
            "status": "skipped",
            "reason": "no_successful_pipeline_records",
            "scores": {},
            "metrics_run": [],
            "metrics_unavailable": ["all"],
        }

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        ContextRelevance,
        Faithfulness,
    )

    from app.rag.evaluation.llm_adapter import build_ragas_embeddings, build_ragas_llm

    has_reference = any("reference" in r for r in rows)
    metrics: List[Any] = [Faithfulness(), AnswerRelevancy(), ContextRelevance()]
    metrics_run = ["faithfulness", "answer_relevancy", "nv_context_relevance"]
    unavailable: List[str] = []

    if has_reference:
        metrics.extend([ContextPrecision(), ContextRecall(), AnswerCorrectness()])
        metrics_run.extend(
            ["context_precision", "context_recall", "answer_correctness"]
        )
    else:
        unavailable.extend(
            ["context_precision", "context_recall", "answer_correctness"]
        )

    # Rows without reference cannot score reference-based metrics; filter
    # the dataset when mixing. Prefer reference subset when available.
    if has_reference:
        eval_rows = [r for r in rows if "reference" in r and r.get("reference")]
    else:
        eval_rows = rows

    if not eval_rows:
        return {
            "status": "skipped",
            "reason": "no_rows_eligible_for_selected_metrics",
            "scores": {},
            "metrics_run": [],
            "metrics_unavailable": metrics_run + unavailable,
        }

    dataset = EvaluationDataset.from_list(eval_rows)
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=build_ragas_llm(llm_provider=llm_provider),
            embeddings=build_ragas_embeddings(),
            raise_exceptions=False,
            show_progress=True,
        )
        scores = summarize_ragas_result(result)
        return {
            "status": "ok",
            "scores": scores,
            "metrics_run": metrics_run,
            "metrics_unavailable": unavailable,
            "n_rows": len(eval_rows),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "scores": {},
            "metrics_run": [],
            "metrics_unavailable": metrics_run + unavailable,
            "n_rows": len(eval_rows),
        }


def aggregate_table_metrics(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "numeric_answer_correct",
        "verified_cell_correct",
        "unit_preserved",
        "page_attribution_ok",
        "table_id_selected",
    ]
    out: Dict[str, Any] = {}
    for key in keys:
        vals = [c.get("table_metrics", {}).get(key) for c in case_results]
        rate = _bool_rate(vals)
        measured = sum(1 for v in vals if isinstance(v, bool))
        out[key] = {
            "accuracy": rate,
            "measured": measured,
            "missing_ground_truth_or_signal": len(vals) - measured,
        }

    failed_table_sel = [
        c.get("id")
        for c in case_results
        if c.get("table_metrics", {}).get("table_id_selected") is False
    ]
    missing_ctx = [
        c.get("id")
        for c in case_results
        if c.get("success") and not (c.get("retrieved_contexts") or [])
    ]
    unsupported = [
        c.get("id")
        for c in case_results
        if c.get("success")
        and any(
            tok in (c.get("response") or "").lower()
            for tok in ("insufficient", "not available", "cannot determine", "ambiguous")
        )
    ]
    out["failed_table_selections"] = failed_table_sel
    out["missing_retrieval_contexts"] = missing_ctx
    out["unsupported_or_abstained_answers"] = unsupported
    return out


def evaluate_dataset(
    dataset_path: Path,
    *,
    user_id: Optional[str] = None,
    llm_provider: Optional[str] = None,
    limit: int = 0,
    skip_llm_metrics: bool = False,
    report_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load dataset → live pipeline → table metrics → RAGAS → report."""
    started = time.perf_counter()
    dataset = load_dataset(dataset_path)
    cases = list(dataset.get("cases") or [])
    if limit and limit > 0:
        cases = cases[:limit]
    uid = str(user_id or dataset.get("user_id") or "").strip()
    if not uid:
        raise ValueError("user_id is required (dataset.user_id or --user-id)")

    case_results: List[Dict[str, Any]] = []
    for case in cases:
        rec = run_live_pipeline_case(
            case, user_id=uid, llm_provider=llm_provider
        )
        table_m = evaluate_table_case(
            case,
            answer=rec.get("response") or "",
            sources=rec.get("sources") or [],
            chunks=rec.get("chunks_meta") or [],
            verified=rec.get("verified_cell"),
        )
        rec["table_metrics"] = table_m
        case_results.append(rec)

    ragas_block = run_ragas_metrics(
        case_results,
        llm_provider=llm_provider,
        skip_llm_metrics=skip_llm_metrics,
    )
    table_block = aggregate_table_metrics(case_results)

    report: Dict[str, Any] = {
        "dataset": dataset.get("name") or Path(dataset_path).stem,
        "dataset_path": str(dataset_path),
        "user_id": uid,
        "n_questions": len(case_results),
        "successful_queries": sum(1 for r in case_results if r.get("success")),
        "failed_queries": sum(1 for r in case_results if not r.get("success")),
        "with_reference": sum(1 for r in case_results if r.get("has_reference")),
        "without_reference": sum(1 for r in case_results if not r.get("has_reference")),
        "retrieval_and_generation_metrics": ragas_block,
        "table_specific_metrics": table_block,
        "evaluation_execution_time_sec": round(time.perf_counter() - started, 3),
        "notes": {
            "fabricated_scores": False,
            "pipeline": "live retrieve → interpret_table_question → generate_answer",
            "null_means": "metric not calculated (missing GT, eval failure, or skip)",
        },
        "cases": case_results,
    }

    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / f"{report['dataset']}_ragas_report.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(out)

        summary_path = report_dir / f"{report['dataset']}_ragas_summary.txt"
        summary_path.write_text(_format_summary(report), encoding="utf-8")
        report["summary_path"] = str(summary_path)

    return report


def merge_overall_report(
    reports: Sequence[Dict[str, Any]],
    *,
    report_dir: Path,
) -> Dict[str, Any]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    overall = {
        "n_datasets": len(reports),
        "n_questions": sum(int(r.get("n_questions") or 0) for r in reports),
        "successful_queries": sum(int(r.get("successful_queries") or 0) for r in reports),
        "failed_queries": sum(int(r.get("failed_queries") or 0) for r in reports),
        "datasets": [
            {
                "name": r.get("dataset"),
                "n_questions": r.get("n_questions"),
                "successful_queries": r.get("successful_queries"),
                "failed_queries": r.get("failed_queries"),
                "ragas": (r.get("retrieval_and_generation_metrics") or {}).get("scores"),
                "ragas_status": (r.get("retrieval_and_generation_metrics") or {}).get(
                    "status"
                ),
                "table_specific_metrics": {
                    k: v
                    for k, v in (r.get("table_specific_metrics") or {}).items()
                    if isinstance(v, dict) and "accuracy" in v
                },
                "report_path": r.get("report_path"),
            }
            for r in reports
        ],
        "notes": {
            "fabricated_scores": False,
            "null_means": "metric not calculated",
        },
    }
    out = report_dir / "overall_ragas_report.json"
    out.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
    overall["report_path"] = str(out)
    return overall


def _format_summary(report: Dict[str, Any]) -> str:
    ragas = report.get("retrieval_and_generation_metrics") or {}
    table = report.get("table_specific_metrics") or {}
    lines = [
        f"Dataset: {report.get('dataset')}",
        f"Questions: {report.get('n_questions')}",
        f"Successful: {report.get('successful_queries')}  Failed: {report.get('failed_queries')}",
        f"With reference: {report.get('with_reference')}  Without: {report.get('without_reference')}",
        f"Execution time (s): {report.get('evaluation_execution_time_sec')}",
        "",
        f"RAGAS status: {ragas.get('status')}",
        f"RAGAS scores: {json.dumps(ragas.get('scores') or {}, indent=2)}",
        f"Metrics unavailable: {ragas.get('metrics_unavailable')}",
        f"RAGAS reason: {ragas.get('reason')}",
        "",
        "Table-specific accuracy (null = not measured):",
    ]
    for key, val in table.items():
        if isinstance(val, dict) and "accuracy" in val:
            lines.append(
                f"  - {key}: accuracy={val.get('accuracy')} "
                f"(measured={val.get('measured')}, "
                f"missing={val.get('missing_ground_truth_or_signal')})"
            )
    lines.append(
        f"Failed table selections: {table.get('failed_table_selections')}"
    )
    lines.append(
        f"Missing retrieval contexts: {table.get('missing_retrieval_contexts')}"
    )
    lines.append(
        f"Unsupported/abstained: {table.get('unsupported_or_abstained_answers')}"
    )
    lines.append("")
    lines.append("No fabricated metric values. null = unavailable.")
    return "\n".join(lines) + "\n"
