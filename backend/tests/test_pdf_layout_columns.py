"""Unit tests for multi-column gutter detection and reading order."""
from __future__ import annotations

from app.rag.ingestion.pdf_layout.analyzer import (
    detect_column_gutter,
    merge_lines_to_blocks,
    order_objects,
    words_to_lines,
)
from app.rag.ingestion.pdf_layout.structure import StructureState, build_page_elements
from app.rag.ingestion.pdf_layout.types import BBox, PageExtraction, TextLine, TextWord


def _word(text: str, x0: float, y0: float, x1: float, y1: float, page: int = 1) -> TextWord:
    return TextWord(
        text=text,
        bbox=BBox(x0, y0, x1, y1),
        page=page,
        confidence=95.0,
    )


def _line(text: str, x0: float, y0: float, x1: float, y1: float, page: int = 1) -> TextLine:
    words = [
        _word(token, x0 + i * 18, y0, x0 + i * 18 + 16, y1, page=page)
        for i, token in enumerate(text.split())
    ]
    if not words:
        words = [_word("", x0, y0, x1, y1, page=page)]
    return TextLine(words=words, page=page)


def test_detect_column_gutter_returns_winning_x_not_loop_end():
    page_width = 595.0
    page_height = 842.0
    lines: list[TextLine] = []
    # Left column ~70-250, right column ~320-520. True gutter near ~285.
    for row in range(12):
        y = 100 + row * 18
        lines.append(_line(f"left column sentence number {row} here", 70, y, 250, y + 12))
        lines.append(_line(f"right column sentence number {row} also", 320, y, 520, y + 12))

    gutter = detect_column_gutter(lines, page_width, page_height)
    assert gutter is not None
    # Must not collapse to the old buggy terminal (~0.72 * width = 428.4).
    assert gutter < page_width * 0.65
    # Synthetic left ends ~176 and right starts at 320.
    assert 176 < gutter < 320


def test_merge_and_order_reads_left_column_then_right():
    page_width = 595.0
    gutter = 285.0
    lines = [
        _line("left one", 70, 100, 240, 112),
        _line("right one", 320, 100, 500, 112),
        _line("left two", 70, 120, 240, 132),
        _line("right two", 320, 120, 500, 132),
        _line("left three", 70, 140, 240, 152),
        _line("right three", 320, 140, 500, 152),
    ]
    blocks = merge_lines_to_blocks(lines, page_width, gutter=gutter, body_size=10.0)
    ordered = order_objects(blocks, gutter, page_width)
    texts = [block.text for block in ordered]
    joined = " ".join(texts)
    assert joined.index("left one") < joined.index("left two") < joined.index("left three")
    assert joined.index("right one") < joined.index("right two") < joined.index("right three")
    assert joined.index("left three") < joined.index("right one")


def test_build_page_elements_does_not_merge_across_columns():
    page_width = 595.0
    page_height = 842.0
    gutter = 285.0
    lines = [
        _line("Management is the most important single factor", 70, 100, 250, 112),
        _line("in a business. Poor management results in", 70, 118, 250, 130),
        _line("grow, assuming he held his many stock", 320, 100, 520, 112),
        _line("dividends, to a peak of one hundred.", 320, 118, 520, 130),
    ]
    page = PageExtraction(
        page_number=1,
        classification="scanned",
        strategy="ocr",
        lines=lines,
        metrics={
            "gutter": gutter,
            "page_width": page_width,
            "page_height": page_height,
        },
    )
    elements = build_page_elements(
        page,
        state=StructureState(),
        body_size=10.0,
        page_width=page_width,
        page_height=page_height,
    )
    paragraphs = [el for el in elements if el.element_type == "paragraph"]
    assert len(paragraphs) >= 2
    left = next(p for p in paragraphs if "Management is the most important" in p.text)
    right = next(p for p in paragraphs if "grow, assuming he held" in p.text)
    assert "grow, assuming" not in left.text
    assert "Management is the most important" not in right.text
    # Reading order: full left column before right column.
    assert elements.index(left) < elements.index(right)


def test_words_to_lines_keeps_cells_separate():
    words = [
        _word("Alpha", 70, 100, 110, 112),
        _word("Beta", 120, 100, 160, 112),
        _word("Gamma", 320, 100, 380, 112),
        _word("Delta", 390, 100, 440, 112),
    ]
    lines = words_to_lines(words, page_width=595.0)
    assert len(lines) >= 2
    assert any("Alpha" in line.text and "Beta" in line.text for line in lines)
    assert any("Gamma" in line.text and "Delta" in line.text for line in lines)
