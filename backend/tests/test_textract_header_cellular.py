"""Header promotion / jammed range-band repair for DetectDocumentText grids."""
from __future__ import annotations

from app.rag.ingestion.pdf_layout.textract_tables import (
    _compose_header_leaf,
    _expand_jammed_header_columns,
    _is_data_row,
    _is_unit_banner_row,
    _leaf_headers,
    _promote_leading_header_bands,
    _split_jammed_range_header,
)


def test_split_jammed_range_header():
    parts = _split_jammed_range_header(
        "≥10-12 >12-15 >15-20 >20-25 >25-30 >30-40 >40-50 >50-60 >60-70"
    )
    assert len(parts) >= 6
    assert parts[0].startswith("≥10") or "10-12" in parts[0]


def test_expand_jammed_fills_column_placeholders():
    headers = [
        "Category",
        "Period",
        "Column 3",
        "Column 4",
        "Column 5",
        "≥10-12 >12-15 >15-20 >20-25 >25-30",
        "Column 7",
    ]
    header_rows = [list(headers)]
    body = [
        [">100-200", "Every 4 Weeks", "75", "75", "75", "150", "150"],
    ]
    hdrs, hrows, out_body = _expand_jammed_header_columns(headers, header_rows, body)
    assert not any(h.lower().startswith("column ") for h in hdrs[2:])
    assert "10-12" in hdrs[2] or "10-12" in hdrs[5]
    assert len(out_body[0]) == len(hdrs)


def test_unit_banner_row_not_data():
    row = [
        "≥30-100",
        "Every 4 Weeks",
        "Amount (mg)",
        "Amount (mg)",
        "Amount (mg)",
        "Amount (mg)",
        "80",
        "150",
    ]
    assert _is_unit_banner_row(row)
    assert not _is_data_row(row)


def test_promote_unit_banner_into_headers():
    headers = ["Key", "Period", "Column 3", "Column 4", "Column 5", "Column 6"]
    header_rows = [list(headers)]
    body = [
        ["", "", "Amount (mg)", "Amount (mg)", "Amount (mg)", "Amount (mg)"],
        [">100-200", "Every 4 Weeks", "75", "150", "150", "300"],
    ]
    leaves, hrows, out_body = _promote_leading_header_bands(headers, header_rows, body)
    assert len(out_body) == 1
    assert out_body[0][0] == ">100-200"
    assert any("Amount" in " ".join(r) or "mg" in " ".join(r).lower() for r in hrows)


def test_compose_header_leaf_joins_parts_as_is():
    leaf = _compose_header_leaf(
        ["Axis Label", "20-25", "kg", "Amount (mg)"]
    )
    assert leaf == "Axis Label 20-25 kg Amount (mg)"


def test_leaf_headers_join_stacked_parts():
    header_rows = [
        ["Row Key", "Period", "Axis Label", "Axis Label"],
        ["", "", "20-25", ">25-30"],
        ["", "", "Amount (mg)", "Amount (mg)"],
    ]
    leaves = _leaf_headers(header_rows)
    assert leaves[2] == "Axis Label 20-25 Amount (mg)"
    assert leaves[3] == "Axis Label >25-30 Amount (mg)"
