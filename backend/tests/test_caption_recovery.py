"""Domain-agnostic caption recovery for empty Docling captions."""
from __future__ import annotations

from app.rag.ingestion.pdf_layout.caption_recovery import (
    enrich_table_caption,
    recover_caption_from_page_blocks,
    recover_caption_from_page_top,
)
from app.rag.ingestion.pdf_layout.types import BBox, TableData
from app.rag.table_rag.understanding import _caption_from_surrounding


def test_recover_caption_from_blocks_above_table():
    # Synthetic page geometry: title above table bbox.
    table_bbox = BBox(72, 160, 540, 500)
    blocks = [
        (72, 90, 500, 105, "Table 1. Widget Rates for Units 12 and Older with Type A"),
        (72, 105, 400, 118, "Continued Title Line"),
        (72, 160, 200, 175, "Pretreatment"),  # grid-ish, must not win
        (72, 180, 200, 200, "30-60 kg"),
    ]
    caption = recover_caption_from_page_blocks(blocks, table_bbox)
    assert caption.startswith("Table 1.")
    assert "Type A" in caption
    assert "Continued Title Line" in caption


def test_recover_prefers_nearest_titled_line():
    table_bbox = BBox(50, 300, 500, 600)
    blocks = [
        (50, 40, 400, 55, "Table 9. Unrelated Older Title Far Above"),
        (50, 250, 480, 280, "Table 2. Pediatric Schedule Ages 6 to <12 Years"),
        (50, 305, 200, 320, "Body Weight"),
    ]
    caption = recover_caption_from_page_blocks(blocks, table_bbox, max_gap_pt=80)
    assert "Table 2." in caption
    assert "Pediatric" in caption
    assert "Table 9." not in caption


def test_enrich_table_caption_noop_when_present():
    td = TableData(
        page=1,
        bbox=BBox(0, 100, 100, 200),
        headers=["A", "B"],
        rows=[["1", "2"]],
        extraction_method="docling",
        caption="Existing Caption",
    )
    out = enrich_table_caption(td, filepath="")
    assert out == "Existing Caption"


def test_caption_from_surrounding_segments():
    surrounding = (
        "Intro prose about the product | "
        "Table 3. Rates for Adult Cohort with Condition X | "
        "Body Weight columns follow"
    )
    caption = _caption_from_surrounding(surrounding)
    assert caption.startswith("Table 3.")
    assert "Condition X" in caption


def test_page_top_caption_when_bbox_unknown():
    blocks = [
        (72, 90, 500, 110, "Table 4. Rates for Cohort Y Ages 1 Year and Older"),
        (72, 130, 200, 150, "Pretreatment"),
        (72, 160, 200, 180, "Body Weight"),
    ]
    caption = recover_caption_from_page_top(blocks)
    assert caption.startswith("Table 4.")
    assert "Cohort Y" in caption
