"""Cross-domain / generalized TableRAG unit tests (no product hardcoding)."""
from __future__ import annotations

from app.rag.generation.structured_lookup import VerifiedCell
from app.rag.generation.table_selector import SELECTOR_SYSTEM_PROMPT
from app.rag.ingestion.pdf_layout.extraction_quality import evaluate_table_quality
from app.rag.ingestion.pdf_layout.types import BBox, TableData
from app.rag.table_rag.routing import QueryModality, route_query


def _table(**kwargs) -> TableData:
    base = dict(
        page=1,
        bbox=BBox(0, 0, 200, 200),
        headers=["A", "B", "C"],
        rows=[["r1", "10", "20"], ["r2", "30", "40"]],
        extraction_method="docling",
        extraction_engine="docling",
        structure_confidence=0.9,
    )
    base.update(kwargs)
    return TableData(**base)


def test_quality_accepts_dense_grid():
    result = evaluate_table_quality(_table())
    assert result.ok
    assert result.structure_confidence >= 0.35


def test_quality_rejects_too_narrow():
    result = evaluate_table_quality(
        _table(headers=["Only"], rows=[["x"]], structure_confidence=0.2)
    )
    assert not result.ok
    assert "too_few_columns" in result.reasons


def test_canonical_to_json_includes_provenance_channels():
    payload = _table(
        ocr_confidence=88.0,
        structure_confidence=0.77,
        provenance={"engine": "docling", "page": 1, "quality_notes": []},
    ).to_json(include_markdown=False)
    assert payload["extraction_engine"] == "docling"
    assert payload["ocr_confidence"] == 88.0
    assert payload["structure_confidence"] == 0.77
    assert payload["provenance"]["engine"] == "docling"


def test_selector_prompt_has_no_product_tokens():
    text = SELECTOR_SYSTEM_PROMPT.lower()
    for banned in ("ige", "xolair", "asthma", "pharmacovigilance", "off-label", "crswnp"):
        assert banned not in text


def test_verified_cell_compare_block_is_domain_agnostic():
    cell = VerifiedCell(
        value="300",
        row_headers=[">300-400"],
        col_headers=[">70-90"],
        row_index=3,
        col_index=4,
        confidence=0.9,
        evidence={"prescribed_value": 150, "prescribed_unit": "mg"},
    )
    block = cell.to_prompt_block().lower()
    assert "verified" in block or "label" in block
    for banned in ("pharmacovigilance", "off-label", "adverse event", "xolair", "ige"):
        assert banned not in block


def test_route_query_axis_pair_is_unit_agnostic():
    # Finance-style: two distinct units (avoid prose openers like "what is").
    route = route_query("FY2024 revenue at 12% growth on a $4.5M base from the table")
    assert route.modality in {QueryModality.TABLE, QueryModality.BOTH}
    # Engineering-style: rpm + mm units.
    route2 = route_query("Table lookup: torque at 2500 rpm for a 12 mm shaft")
    assert route2.needs_tables or route2.modality == QueryModality.BOTH


def test_finance_and_engineering_grids_are_quality_ok():
    finance = _table(
        headers=["Year", "Revenue ($M)", "Expense ($M)"],
        rows=[["2023", "12.4", "8.1"], ["2024", "14.0", "9.2"]],
    )
    engineering = _table(
        headers=["Material", "Yield (MPa)", "UTS (MPa)"],
        rows=[["A36", "250", "400"], ["A572", "345", "450"]],
    )
    assert evaluate_table_quality(finance).ok
    assert evaluate_table_quality(engineering).ok
