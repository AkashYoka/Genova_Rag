"""Benchmark harness unit tests (fixture mode — no live PDF required)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "bench_table_extraction",
    _ROOT / "scripts" / "bench_table_extraction.py",
)
_bench = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_bench)
cell_metrics = _bench.cell_metrics
run_fixture_cases = _bench.run_fixture_cases


def test_cell_metrics_perfect_match():
    expected = {"headers": ["A", "B"], "rows": [["1", "2"]]}
    actual = {"headers": ["A", "B"], "rows": [["1", "2"]], "extraction_engine": "docling"}
    m = cell_metrics(expected, actual)
    assert m["header_correct"] is True
    assert m["cell_accuracy"] == 1.0
    assert m["row_col_relationship_ok"] is True


def test_cell_metrics_detects_header_and_cell_errors():
    expected = {"headers": ["A", "B"], "rows": [["1", "2"], ["3", "4"]]}
    actual = {"headers": ["A", "X"], "rows": [["1", "9"], ["3", "4"]]}
    m = cell_metrics(expected, actual)
    assert m["header_correct"] is False
    assert m["cell_accuracy"] < 1.0
    assert m["cells_matched"] == 3
    assert m["cells_total"] == 4


def test_fixture_dir_scores_cross_domain_cases():
    fixture_dir = Path(__file__).resolve().parents[1] / "data" / "evals" / "table_extraction"
    cases = []
    for path in sorted(fixture_dir.glob("*.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    assert len(cases) >= 3
    report = run_fixture_cases(cases)
    assert report["n_scored"] >= 3
    assert report["header_accuracy"] == 1.0
    assert report["mean_cell_accuracy"] == 1.0
