"""Figure/chart OCR grids must not become document_tables."""
from __future__ import annotations

from app.rag.ingestion.pdf_layout.extraction_quality import (
    evaluate_table_quality,
    is_figure_like_table,
)
from app.rag.ingestion.pdf_layout.types import BBox, TableData


def test_figure_on_page_caption_is_rejected():
    table = TableData(
        page=43,
        bbox=BBox(0, 0, 100, 100),
        headers=["Column 1", "Column 2", "Column 3", "Column 4"],
        rows=[["-1.6", "-0.7", "0.2", "Baseline"]],
        caption="Figure on page 43",
    )
    assert is_figure_like_table(table) is True
    result = evaluate_table_quality(table)
    assert result.ok is False
    assert "figure_like_table" in result.reasons


def test_real_dosing_table_caption_is_kept():
    table = TableData(
        page=5,
        bbox=BBox(0, 0, 100, 100),
        headers=[
            "Pretreatment Serum IgE (IU/mL)",
            "Dosing Freq.",
            "Body Weight / >70-90 kg / Dose (mg)",
        ],
        rows=[[">300-400", "Every 2 weeks", "300"]],
        caption=(
            "Table 1. Subcutaneous XOLAIR Doses for Patients 12 Years "
            "of Age and Older with Asthma"
        ),
    )
    assert is_figure_like_table(table) is False
    result = evaluate_table_quality(table)
    assert result.ok is True
