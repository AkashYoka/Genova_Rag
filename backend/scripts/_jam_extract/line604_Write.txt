"""Header promotion / jammed weight-band repair for DetectDocumentText grids."""
from __future__ import annotations

from app.rag.ingestion.pdf_layout.textract_tables import (
    _compose_weight_dose_leaf,
    _expand_jammed_header_columns,
    _is_data_row,
    _is_unit_dose_row,
    _leaf_headers,
    _promote_leading_header_bands,
    _split_jammed_weight_header,
)


def test_split_jammed_weight_header():
    parts = _split_jammed_weight_header(
        "≥10-12 >12-15 >15-20 >20-25 >25-30 >30-40 >40-50 >50-60 >60-70"
    )
    assert len(parts) >= 6
    assert parts[0].startswith("≥10") or "10-12" in parts[0]


def test_expand_jammed_fills_column_placeholders():
    headers = [
        "Pretreatment Serum IgE (IU/mL)",
        "Dosing Freq.",
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


def test_unit_dose_row_not_data():
    row = [
        "≥30-100",
        "Every 4 Weeks",
        "Dose (mg)",
        "Dose (mg)",
        "Dose (mg)",
        "Dose (mg)",
        "80",
        "150",
    ]
    assert _is_unit_dose_row(row)
    assert not _is_data_row(row)


def test_promote_dose_banner_into_headers():
    headers = ["IgE", "Freq", "Column 3", "Column 4", "Column 5", "Column 6"]
    header_rows = [list(headers)]
    body = [
        ["", "", "Dose (mg)", "Dose (mg)", "Dose (mg)", "Dose (mg)"],
        [">100-200", "Every 4 Weeks", "75", "150", "150", "300"],
    ]
    leaves, hrows, out_body = _promote_leading_header_bands(headers, header_rows, body)
    assert len(out_body) == 1
    assert out_body[0][0] == ">100-200"
    assert any("Dose" in " ".join(r) or "mg" in " ".join(r).lower() for r in hrows)


def test_compose_weight_dose_leaf():
    leaf = _compose_weight_dose_leaf(
        ["Body Weight", "20-25", "kg", "Dose (mg)"]
    )
    assert leaf == "Body Weight 20-25 kg Dose (mg)"


def test_leaf_headers_compose_weight_dose():
    header_rows = [
        ["Pretreatment Serum IgE (IU/mL)", "Dosing Freq.", "Body Weight", "Body Weight"],
        ["", "", "20-25", ">25-30"],
        ["", "", "Dose (mg)", "Dose (mg)"],
    ]
    leaves = _leaf_headers(header_rows)
    assert leaves[2] == "Body Weight 20-25 kg Dose (mg)"
    assert leaves[3] == "Body Weight >25-30 kg Dose (mg)"
