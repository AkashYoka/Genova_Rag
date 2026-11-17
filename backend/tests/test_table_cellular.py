"""Fully-cellular table reconstruction (domain-agnostic / structural)."""
from __future__ import annotations

from app.rag.ingestion.pdf_layout.table_cellular import (
    collapse_repeated_phrase,
    expand_to_fully_cellular,
    table_data_to_html,
)
from app.rag.ingestion.pdf_layout.types import BBox, TableData


def test_shared_categorical_attr_bidirectional_fill():
    headers = ["Key", "Cadence", "A", "B"]
    rows = [
        [">100-200", "", "75", "75"],
        [">200-300", "", "150", "150"],
        [">300-400", "Every 4 weeks", "150", "225"],
        [">700-800", "Every 2 weeks", "225", "300"],
        [">800-900", "", "300", "375"],
    ]
    out = expand_to_fully_cellular(headers, rows)
    cadences = [r[1] for r in out["rows"]]
    assert cadences == [
        "Every 4 weeks",
        "Every 4 weeks",
        "Every 4 weeks",
        "Every 2 weeks",
        "Every 2 weeks",
    ]
    # Headers must not be flooded/rewritten.
    assert out["headers"] == headers
    assert out["merged_cells"] == []


def test_collapse_repeated_phrase():
    phrase = "Insufficient Data to Recommend a Dose"
    spam = " ".join([phrase] * 4)
    assert collapse_repeated_phrase(spam) == "Insufficient Data"
    assert collapse_repeated_phrase(phrase) == "Insufficient Data"
    assert collapse_repeated_phrase("Monthly Monthly") == "Monthly"


def test_marker_region_dedupes_and_expands():
    headers = ["Row", "A", "B", "C", "D"]
    phrase = "Insufficient Data to Recommend a Dose"
    spam = " ".join([phrase] * 4)
    rows = [
        ["r1", "10", "20", "30", "40"],
        ["r2", "10", "20", spam, spam],
        ["r3", "10", "", "", ""],
        ["r4", "", "", "", ""],
    ]
    out = expand_to_fully_cellular(headers, rows)
    body = out["rows"]
    assert body[1][3] == "Insufficient Data"
    assert body[1][4] == "Insufficient Data"
    assert body[2][3] == "Insufficient Data"
    assert body[2][4] == "Insufficient Data"
    assert body[3][3] == "Insufficient Data"
    assert body[3][4] == "Insufficient Data"
    assert body[2][1] == "10"
    assert out["headers"] == headers


def test_financial_rate_card_domain_agnostic():
    headers = ["Tier", "Billing Cadence", "USD", "EUR", "GBP"]
    rows = [
        ["Starter", "Monthly", "9", "8", "7"],
        ["Growth", "", "29", "N/A", ""],
        ["Enterprise", "Annual", "99", "89", "79"],
        ["Custom", "", "", "not applicable", ""],
    ]
    out = expand_to_fully_cellular(headers, rows)
    assert out["rows"][1][1] == "Monthly"
    assert out["rows"][3][1] == "Annual"
    assert out["rows"][1][3] == "N/A"
    assert out["rows"][1][4] == "N/A"
    assert out["rows"][3][3] == "not applicable"
    assert out["rows"][3][4] == "not applicable"
    assert out["headers"] == headers


def test_html_has_no_rowspan_or_colspan():
    headers = ["A", "B"]
    rows = [["1", "N/A"], ["2", "3"]]
    html = table_data_to_html(headers, rows, caption="Demo")
    assert "<table" in html
    assert "rowspan" not in html.lower()
    assert "colspan" not in html.lower()


def test_tabledata_json_includes_html_and_empty_merges():
    table = TableData(
        page=1,
        bbox=BBox(0, 0, 100, 100),
        headers=["H1", "H2"],
        rows=[["a", "N/A"]],
        merged_cells=[{"row_index": 0, "col_index": 0, "row_span": 2, "col_span": 2}],
    )
    data = table.to_json()
    assert data["merged_cells"] == []
    assert "<table" in (data.get("html") or "")
