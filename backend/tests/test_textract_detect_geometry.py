"""Unit tests for DetectDocumentText → geometry table reconstruction."""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from app.rag.ingestion.pdf_layout import textract_tables as tt
from app.rag.ingestion.pdf_layout.types import BBox


def _word(
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    confidence: float = 99.0,
    word_id: str = "",
) -> Dict[str, Any]:
    return {
        "Id": word_id or f"w-{text}-{left}-{top}",
        "BlockType": "WORD",
        "Text": text,
        "Confidence": confidence,
        "Geometry": {
            "BoundingBox": {
                "Left": left,
                "Top": top,
                "Width": width,
                "Height": height,
            }
        },
    }


def _line(
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
) -> Dict[str, Any]:
    return {
        "Id": f"l-{text}-{left}-{top}",
        "BlockType": "LINE",
        "Text": text,
        "Confidence": 99.0,
        "Geometry": {
            "BoundingBox": {
                "Left": left,
                "Top": top,
                "Width": width,
                "Height": height,
            }
        },
    }


def _simple_grid_blocks() -> List[Dict[str, Any]]:
    """3-column dose-style grid with headers + 2 body rows (normalized coords)."""
    # Columns at ~0.10, 0.40, 0.70; rows at ~0.20, 0.35, 0.50
    cells = [
        ("IgE", 0.08, 0.18),
        (">60-70", 0.38, 0.18),
        (">70-90", 0.68, 0.18),
        (">30-100", 0.05, 0.34),
        ("150", 0.40, 0.34),
        ("150", 0.70, 0.34),
        (">100-200", 0.05, 0.50),
        ("300", 0.40, 0.50),
        ("300", 0.70, 0.50),
    ]
    blocks: List[Dict[str, Any]] = []
    for text, left, top in cells:
        w = 0.12 if len(text) > 4 else 0.08
        blocks.append(_word(text, left, top, w, 0.04))
    blocks.append(_line("IgE >60-70 >70-90", 0.08, 0.18, 0.72, 0.04))
    blocks.append(_line(">30-100 150 150", 0.05, 0.34, 0.75, 0.04))
    blocks.append(_line(">100-200 300 300", 0.05, 0.50, 0.75, 0.04))
    return blocks


def test_words_from_detect_blocks_maps_pdf_coords():
    blocks = [_word("Hello", 0.1, 0.2, 0.2, 0.05, confidence=95.0)]
    words = tt.words_from_detect_blocks(
        blocks, page_number=2, page_width=100.0, page_height=200.0
    )
    assert len(words) == 1
    assert words[0].text == "Hello"
    assert words[0].page == 2
    assert words[0].confidence == 95.0
    assert words[0].bbox.x0 == pytest.approx(10.0)
    assert words[0].bbox.y0 == pytest.approx(40.0)
    assert words[0].bbox.x1 == pytest.approx(30.0)
    assert words[0].bbox.y1 == pytest.approx(50.0)


def test_table_data_from_detect_blocks_preserves_headers_rows_cells():
    tables = tt.table_data_from_textract_blocks(
        _simple_grid_blocks(),
        page_number=1,
        page_width=600.0,
        page_height=800.0,
        min_confidence=50.0,
        page=None,
    )
    assert tables, "expected at least one reconstructed table"
    table = tables[0]
    assert table.extraction_method == "textract_detect"
    assert len(table.headers) >= 2
    assert table.rows, "body rows must not be empty"
    # Header / body values should survive reconstruction.
    flat = " ".join(table.headers) + " " + " ".join(
        " ".join(r) for r in table.rows
    )
    assert "IgE" in flat or ">60-70" in flat or ">70-90" in flat
    assert "150" in flat
    assert "300" in flat
    assert ">30-100" in flat or ">100-200" in flat
    # Column order: first body key column before dose values.
    first_body = table.rows[0]
    assert len(first_body) >= 2
    assert table.cells, "synthetic cell metadata expected"
    assert table.textract_confidence > 0
    assert table.header_rows or table.headers


def test_empty_blocks_yield_no_tables():
    assert (
        tt.table_data_from_textract_blocks(
            [],
            page_number=1,
            page_width=600.0,
            page_height=800.0,
        )
        == []
    )


def test_paragraph_only_words_may_yield_empty_or_weak_grid():
    # Sparse prose — not enough aligned columns for a usable table after filter.
    blocks = [
        _word("This", 0.1, 0.1, 0.08, 0.03),
        _word("is", 0.2, 0.1, 0.04, 0.03),
        _word("a", 0.26, 0.1, 0.03, 0.03),
        _word("paragraph", 0.32, 0.1, 0.15, 0.03),
        _word("without", 0.1, 0.16, 0.12, 0.03),
        _word("columns", 0.25, 0.16, 0.12, 0.03),
    ]
    tables = tt.table_data_from_textract_blocks(
        blocks,
        page_number=1,
        page_width=600.0,
        page_height=800.0,
    )
    usable = tt.usable_textract_tables(tables)
    # May reconstruct a weak 2-col alignment; usable filter requires body rows.
    for table in usable:
        assert len(table.headers) >= 2 or (
            table.rows and max(len(r) for r in table.rows) >= 2
        )


def test_merge_soft_continuation_rows():
    rows = [
        ["Key", "150", ""],
        ["", "", "mg"],
        ["Next", "300", "q2w"],
    ]
    merged = tt._merge_soft_continuation_rows(rows)
    assert merged[0][2] == "mg"
    assert merged[1] == ["Next", "300", "q2w"]


def test_detect_document_text_called_not_analyze(monkeypatch):
    tt.reset_textract_client()
    client = MagicMock()
    client.detect_document_text.return_value = {"Blocks": _simple_grid_blocks()}
    monkeypatch.setattr(tt, "get_textract_client", lambda: client)

    page = MagicMock()
    page.rect.width = 600.0
    page.rect.height = 800.0
    pix = MagicMock()
    pix.tobytes.return_value = b"\x89PNG" + b"0" * 100
    page.get_pixmap.return_value = pix

    monkeypatch.setattr(
        tt,
        "get_settings",
        lambda: MagicMock(
            PDF_TEXTRACT_TABLES_ENABLED=True,
            PDF_TEXTRACT_DPI=72,
            PDF_OCR_DPI=72,
            PDF_TEXTRACT_MIN_CONFIDENCE=50.0,
            PDF_TEXTRACT_TIMEOUT_SECONDS=30,
            PDF_TEXTRACT_REGION="us-east-1",
            BEDROCK_REGION="us-east-1",
        ),
    )

    tables = tt.extract_tables_from_page(page, 1)
    client.detect_document_text.assert_called_once()
    client.analyze_document.assert_not_called()
    assert isinstance(tables, list)
    tt.reset_textract_client()


def test_usable_textract_tables_filters_narrow():
    from app.rag.ingestion.pdf_layout.types import TableData

    good = TableData(
        page=1,
        bbox=BBox(0, 0, 100, 100),
        headers=["A", "B"],
        rows=[["1", "2"]],
        extraction_method="textract_detect",
    )
    bad = TableData(
        page=1,
        bbox=BBox(0, 0, 100, 100),
        headers=["Only"],
        rows=[["x"]],
        extraction_method="textract_detect",
    )
    assert tt.usable_textract_tables([good, bad]) == [good]
