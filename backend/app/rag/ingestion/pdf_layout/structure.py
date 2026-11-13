"""Document structure reconstruction: headings, sections, lists, captions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.rag.table_rag.ids import make_table_id

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
# Prescribing-info / manual TOCs: "1 INDICATIONS AND USAGE ........ 3"
_TOC_DOT_LEADER_RE = re.compile(r"\.{2,}\s*\d{1,4}\s*$")
_TOC_NUMBERED_PAGE_RE = re.compile(
    r"^(\d+(?:\.\d+){0,3})\s+(.+?)\s+(\d{1,4})\s*$"
)
_CONTENTS_TITLE_RE = re.compile(
    r"(?i)^(?:(?:full\s+)?prescribing\s+information:\s*)?contents\*?\s*$"
    r"|^\s*table\s+of\s+contents\s*$"
)
_TOC_EXIT_RE = re.compile(
    r"(?i)^full\s+prescribing\s+information\s*$"
    r"|^warning:\s*"
    r"|^\d+(?:\.\d+)*\s+[A-Za-z].{2,}$"
)


@dataclass
class StructureState:
    section: str = ""
    subsection: str = ""
    heading_stack: List[str] = field(default_factory=list)
    # When True, numbered TOC listing lines must not become section headings.
    in_toc: bool = False


def looks_like_toc_entry(text: str) -> bool:
    """True for TOC listing lines (title + page), not real body headings."""
    t = normalize_ws(text or "")
    if not t or len(t) > 160:
        return False
    if _TOC_DOT_LEADER_RE.search(t):
        return True
    match = _TOC_NUMBERED_PAGE_RE.match(t)
    if not match:
        return False
    title = match.group(2).strip()
    if _SENTENCE_END_RE.search(title):
        return False
    # Real body headings rarely end with a bare page number token.
    return len(title) <= 110


def is_contents_title(text: str) -> bool:
    return bool(_CONTENTS_TITLE_RE.match(normalize_ws(text or "")))


def is_toc_exit_heading(text: str) -> bool:
    """Body start after a CONTENTS listing (e.g. FULL PRESCRIBING INFORMATION)."""
    t = normalize_ws(text or "")
    if not t or looks_like_toc_entry(t) or is_contents_title(t):
        return False
    if re.match(r"(?i)^full\s+prescribing\s+information\s*$", t):
        return True
    if re.match(r"(?i)^warning:\s*", t):
        return True
    # Numbered section heading without trailing page number.
    return bool(_NUMBERED_HEADING_RE.match(t) and not _TOC_NUMBERED_PAGE_RE.match(t))


def body_font_size(pages: Iterable[PageExtraction]) -> float:
    sizes: List[float] = []
    for page in pages:
        for line in page.lines:
            if len(line.words) >= 3 and line.height > 0:
                sizes.append(line.height)
    return median(sizes) or 11.0


def heading_level(line: TextLine, body_size: float, *, in_toc: bool = False) -> int:
    """Return 1/2/3 for a heading, or 0 for body text."""
    text = line.text.strip()
    if not text or len(text) > 180:
        return 0
    # Inside a CONTENTS listing, only the CONTENTS title itself may be a heading.
    if in_toc and not is_contents_title(text):
        if looks_like_toc_entry(text) or _NUMBERED_HEADING_RE.match(text):
            return 0
    if looks_like_toc_entry(text):
        return 0
    size = line.font_size or line.height
    ratio = size / body_size if body_size else 1.0
    if is_contents_title(text):
        return 1
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
    if is_contents_title(heading):
        state.in_toc = True
        state.section = "Table of Contents"
        state.subsection = ""
        state.heading_stack = ["Table of Contents"]
        return
    if state.in_toc and is_toc_exit_heading(heading):
        state.in_toc = False
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
    """Nearest caption above or below a table/figure (layout-generic).

    Multi-line table titles (``Table N. ...`` wrapped onto the next line) are
    merged so applicability cues such as age bands are not truncated.
    """
    threshold = max(36.0, page_height * 0.06)
    candidates: List[tuple[float, TextBlock]] = []
    for block in blocks:
        if not is_caption_line(block.text):
            continue
        if not (block.bbox.x0 < bbox.x1 and block.bbox.x1 > bbox.x0):
            continue
        gap = min(
            abs(block.bbox.y0 - bbox.y1),  # caption below the object
            abs(bbox.y0 - block.bbox.y1),  # caption above the object
        )
        if gap <= threshold:
            candidates.append((gap, block))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    primary = candidates[0][1]
    # Merge immediately following non-caption continuation lines.
    ordered = sorted(
        [b for b in blocks if b.text],
        key=lambda b: (b.bbox.y0, b.bbox.x0),
    )
    try:
        idx = next(i for i, b in enumerate(ordered) if b is primary)
    except StopIteration:
        return primary
    parts = [primary.text]
    y_limit = primary.bbox.y1 + max(28.0, page_height * 0.035)
    for nxt in ordered[idx + 1 : idx + 3]:
        if is_caption_line(nxt.text):
            break
        if nxt.bbox.y0 > y_limit:
            break
        if not (nxt.bbox.x0 < bbox.x1 and nxt.bbox.x1 > bbox.x0):
            break
        # Continuation: short line without starting a new section heading style.
        text = normalize_ws(nxt.text)
        if not text or len(text) > 160:
            break
        parts.append(text)
        y_limit = nxt.bbox.y1 + max(22.0, page_height * 0.03)
    if len(parts) == 1:
        return primary
    merged = TextBlock(
        lines=list(primary.lines) + [
            line for b in ordered[idx + 1 : idx + len(parts)] for line in b.lines
        ],
        page=primary.page,
        column=primary.column,
        block_type=primary.block_type,
        bbox_override=BBox(
            min(primary.bbox.x0, *(ordered[idx + i].bbox.x0 for i in range(len(parts) - 1))),
            primary.bbox.y0,
            max(primary.bbox.x1, *(ordered[idx + i].bbox.x1 for i in range(len(parts) - 1))),
            max(primary.bbox.y1, *(ordered[idx + i].bbox.y1 for i in range(len(parts) - 1))),
        ),
    )
    return merged


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

    table_ordinal = 0
    for obj in ordered:
        if isinstance(obj, TableData):
            flush_paragraph()
            flush_list()
            caption = _caption_near(blocks, obj.bbox, page_height)
            if caption:
                obj.caption = caption.text
            table_payload = obj.to_json()
            # Stable id so TableRAG SQL + row/summary index chunks can
            # reference the same extracted grid across split parts.
            table_payload["table_id"] = make_table_id(
                page=int(obj.page or 1),
                ordinal=table_ordinal,
                bbox=obj.bbox.as_tuple() if obj.bbox else None,
                headers=obj.headers or [],
            )
            table_ordinal += 1
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
                        "table": table_payload,
                        "rows": obj.n_rows,
                        "columns": obj.n_cols,
                        "caption": obj.caption,
                        "table_id": table_payload["table_id"],
                    },
                )
            )
            continue

        block: TextBlock = obj
        # Enter TOC mode from the CONTENTS title even when font sizing is weak.
        if is_contents_title(block.text):
            state.in_toc = True
        elif state.in_toc and is_toc_exit_heading(block.text):
            state.in_toc = False

        level = heading_level(block.lines[0], body_size, in_toc=state.in_toc)
        if level:
            flush_paragraph()
            flush_list()
            heading = block.text
            # Normalize the CONTENTS banner so downstream hierarchy is stable.
            if is_contents_title(heading):
                heading = "Table of Contents"
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

        # TOC listing lines stay paragraphs so they cannot pollute section state.
        if state.in_toc and looks_like_toc_entry(block.text):
            flush_list()
            paragraph.extend(block.lines)
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
