"""Convert layout-extracted document elements into stable source blocks."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

from app.rag.chunking.semantic.types import SourceBlock
from app.rag.ingestion.pdf_layout.types import DocumentElement, normalize_ws

_HEADING_MARKER_RE = re.compile(r"^#{1,6}\s+")


def _clean_heading(value: str) -> str:
    return normalize_ws(_HEADING_MARKER_RE.sub("", value or ""))


def _table_payload(element: DocumentElement) -> Dict[str, Any]:
    table = element.metadata.get("table")
    if isinstance(table, dict):
        return dict(table)
    return {}


def _block_text(element: DocumentElement, table: Dict[str, Any]) -> str:
    text = (element.text or "").strip()
    if element.element_type == "table":
        markdown = str(table.get("markdown") or "").strip()
        # Layout tables carry markdown + linearized facts in ``element.text``.
        # Prefer the richer text, falling back to the structured markdown.
        if text:
            return text
        return markdown
    return text


def extract_source_blocks(elements: Sequence[DocumentElement]) -> List[SourceBlock]:
    """Return ordered, immutable blocks with deterministic ``b00001`` ids.

    Empty elements are skipped. Block ids are contiguous over kept blocks so a
    reindex of identical content produces identical ids and section ids.
    """
    blocks: List[SourceBlock] = []
    for element in elements:
        table = _table_payload(element) if element.element_type == "table" else {}
        text = _block_text(element, table)
        if not text.strip():
            continue

        element_type = element.element_type or "paragraph"
        if element_type == "heading":
            text = _clean_heading(text)
            if not text:
                continue

        bbox = list(element.bbox.as_tuple()) if element.bbox is not None else None
        block = SourceBlock(
            block_id=f"b{len(blocks) + 1:05d}",
            order=len(blocks),
            element_type=element_type,
            text=text,
            page_start=int(element.page_start or 1),
            page_end=int(element.page_end or element.page_start or 1),
            level=int(element.level or 0),
            section=element.section or "",
            subsection=element.subsection or "",
            extraction_method=element.extraction_method or "native",
            content_type=element.content_type or "paragraph",
            bbox=bbox,
            table=table or None,
        )
        blocks.append(block)
    return blocks


def blocks_by_id(blocks: Iterable[SourceBlock]) -> Dict[str, SourceBlock]:
    return {block.block_id: block for block in blocks}


def page_span(blocks: Iterable[SourceBlock]) -> tuple[int, int]:
    pages: List[int] = []
    for block in blocks:
        pages.append(int(block.page_start or 1))
        pages.append(int(block.page_end or block.page_start or 1))
    if not pages:
        return 1, 1
    return min(pages), max(pages)
