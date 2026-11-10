"""Document structure reconstruction: headings, sections, lists, captions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.rag.ingestion.pdf_layout.analyzer import (
    is_caption_line,
    is_list_line,
    median,
    order_objects,
)
from app.rag.ingestion.pdf_layout.types import (
    BBox,
    DocumentElement,
    PageExtraction,
    TableData,
    TextBlock,
    TextLine,
    normalize_ws,
)

_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\s+\S")
_CHAPTER_RE = re.compile(r"^(chapter|appendix|part|section)\b", re.I)
_SENTENCE_END_RE = re.compile(r"[.!?;:]$")


@dataclass
class StructureState:
    section: str = ""
    subsection: str = ""
    heading_stack: List[str] = field(default_factory=list)


def body_font_size(pages: Iterable[PageExtraction]) -> float:
    sizes: List[float] = []
    for page in pages:
        for line in page.lines:
            if len(line.words) >= 3 and line.height > 0:
                sizes.append(line.height)
    return median(sizes) or 11.0


def heading_level(line: TextLine, body_size: float) -> int:
    """Return 1/2/3 for a heading, or 0 for body text."""
    text = line.text.strip()
    if not text or len(text) > 180:
        return 0
    size = line.font_size or line.height
    ratio = size / body_size if body_size else 1.0
    if _CHAPTER_RE.match(text):
        return 1
    numbered = _NUMBERED_HEADING_RE.match(text)
    if numbered and len(text) <= 110:
        return min(3, numbered.group(1).count(".") + 1)
    if ratio >= 1.55:
        return 1
    if ratio >= 1.22:
        return 2
    letters = [c for c in text if c.isalpha()]
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    if upper_ratio >= 0.85 and len(text) <= 90 and not _SENTENCE_END_RE.search(text):
        return 2
    if line.bold and ratio >= 1.05 and len(text) <= 120:
        return 3
    return 0


def _update_state(state: StructureState, heading: str, level: int) -> None:
    if level <= 1:
        state.section = heading
        state.subsection = ""
        state.heading_stack = [heading]
    elif level == 2:
        state.subsection = heading
        state.heading_stack = state.heading_stack[:1] + [heading]
    else:
        state.subsection = heading
        state.heading_stack = state.heading_stack[:2] + [heading]


def _join_lines(lines: Sequence[TextLine]) -> str:
    parts: List[str] = []
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        if parts and parts[-1].endswith("-"):
            parts[-1] = parts[-1][:-1] + text
        else:
            parts.append(text)
    return normalize_ws(" ".join(parts))


def _caption_near(blocks: Sequence[TextBlock], bbox: BBox, page_height: float) -> Optional[TextBlock]:
    candidates = [
        block for block in blocks
        if is_caption_line(block.text)
        and abs(block.bbox.y0 - bbox.y1) <= max(24.0, page_height * 0.03)
        and block.bbox.x0 < bbox.x1
        and block.bbox.x1 > bbox.x0
    ]
    return candidates[0] if candidates else None


def build_page_elements(
    page: PageExtraction,
    *,
    state: StructureState,
    body_size: float,
    page_width: float,
    page_height: float,
    figures: Optional[Sequence[DocumentElement]] = None,
) -> List[DocumentElement]:
    """Convert ordered lines/tables into structured, section-aware elements."""
    gutter = page.metrics.get("gutter")
    blocks = [
        block for block in _lines_to_blocks(page.lines, page_width, gutter, body_size)
        if block.text
    ]
    caption_lines = [block for block in blocks if is_caption_line(block.text)]
    table_objects: List[Any] = list(page.tables)
    objects: List[Any] = blocks + table_objects
    ordered = order_objects(objects, gutter, page_width)

    elements: List[DocumentElement] = []
    paragraph: List[TextLine] = []
    paragraph_column: Optional[int] = None
    list_lines: List[TextLine] = []

    def flush_paragraph() -> None:
        nonlocal paragraph, paragraph_column
        if not paragraph:
            return
        text = _join_lines(paragraph)
        if text:
            column_ids = sorted(
                {
                    (1 if (gutter is not None and line.bbox.center_x > gutter) else 0)
                    for line in paragraph
                }
            )
            elements.append(
                DocumentElement(
                    element_type="paragraph",
                    text=text,
                    page_start=paragraph[0].page,
                    page_end=paragraph[-1].page,
                    bbox=_union([line.bbox for line in paragraph]),
                    section=state.section,
                    subsection=state.subsection,
                    extraction_method="ocr" if paragraph[0].source == "ocr" else "native",
                    content_type="paragraph",
                    metadata={
                        "ocr_confidence": round(
                            sum(line.confidence for line in paragraph) / len(paragraph), 2
                        ),
                        "columns": column_ids,
                    },
                )
            )
        paragraph = []
        paragraph_column = None

    def flush_list() -> None:
        nonlocal list_lines
        if not list_lines:
            return
        text = "\n".join(line.text for line in list_lines if line.text)
        if text:
            elements.append(
                DocumentElement(
                    element_type="list",
                    text=text,
                    page_start=list_lines[0].page,
                    page_end=list_lines[-1].page,
                    bbox=_union([line.bbox for line in list_lines]),
                    section=state.section,
                    subsection=state.subsection,
                    extraction_method="ocr" if list_lines[0].source == "ocr" else "native",
                    content_type="list",
                    metadata={"items": len(list_lines)},
                )
            )
        list_lines = []

    for obj in ordered:
        if isinstance(obj, TableData):
            flush_paragraph()
            flush_list()
            caption = _caption_near(blocks, obj.bbox, page_height)
            if caption:
                obj.caption = caption.text
            elements.append(
                DocumentElement(
                    element_type="table",
                    text=obj.markdown,
                    page_start=obj.page,
                    page_end=obj.page,
                    bbox=obj.bbox,
                    section=state.section,
                    subsection=state.subsection,
                    extraction_method=obj.extraction_method,
                    content_type="table",
                    metadata={
                        "table": obj.to_json(),
                        "rows": obj.n_rows,
                        "columns": obj.n_cols,
                        "caption": obj.caption,
                    },
                )
            )
            continue

        block: TextBlock = obj
        level = heading_level(block.lines[0], body_size)
        if level:
            flush_paragraph()
            flush_list()
            heading = block.text
            _update_state(state, heading, level)
            elements.append(
                DocumentElement(
                    element_type="heading",
                    text=heading,
                    page_start=block.page,
                    page_end=block.page,
                    bbox=block.bbox,
                    level=level,
                    section=state.section,
                    subsection=state.subsection,
                    extraction_method="ocr" if block.lines[0].source == "ocr" else "native",
                    content_type="heading",
                    metadata={"font_size": round(block.font_size, 2)},
                )
            )
            continue

        if is_caption_line(block.text):
            flush_paragraph()
            flush_list()
            elements.append(
                DocumentElement(
                    element_type="caption",
                    text=block.text,
                    page_start=block.page,
                    page_end=block.page,
                    bbox=block.bbox,
                    section=state.section,
                    subsection=state.subsection,
                    extraction_method="ocr" if block.lines[0].source == "ocr" else "native",
                    content_type="figure_caption",
                    metadata={},
                )
            )
            continue

        if is_list_line(block.text):
            flush_paragraph()
            list_lines.extend(block.lines)
            continue

        flush_list()
        block_column = getattr(block, "column", None)
        if block_column is None and gutter is not None and block.bbox is not None:
            block_column = 1 if block.bbox.center_x > gutter else 0
        if (
            paragraph
            and paragraph_column is not None
            and block_column is not None
            and block_column != paragraph_column
        ):
            flush_paragraph()
        if paragraph_column is None:
            paragraph_column = block_column
        paragraph.extend(block.lines)

    flush_paragraph()
    flush_list()

    # Insert figures by vertical position without destroying column order via a
    # global Y-only re-sort of every element.
    for figure in figures or []:
        insert_at = len(elements)
        fig_y = figure.bbox.y0 if figure.bbox else 0.0
        for index, element in enumerate(elements):
            if element.page_start > figure.page_start:
                insert_at = index
                break
            if element.page_start == figure.page_start:
                elem_y = element.bbox.y0 if element.bbox else 0.0
                if elem_y > fig_y:
                    insert_at = index
                    break
        elements.insert(insert_at, figure)
    return elements


def _lines_to_blocks(
    lines: Sequence[TextLine],
    page_width: float,
    gutter: Optional[float],
    body_size: float,
) -> List[TextBlock]:
    from app.rag.ingestion.pdf_layout.analyzer import merge_lines_to_blocks

    return merge_lines_to_blocks(lines, page_width, gutter, body_size)


def _union(boxes: Sequence[BBox]) -> Optional[BBox]:
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return BBox(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )
