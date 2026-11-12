"""Reconstruct retrieval-ready chunks from original source blocks.

The LLM never writes chunk text. Sections only provide boundaries; content is
always rendered from the extracted blocks, preserving tables, page ranges and
source block ids.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.rag.chunking.semantic.blocks import blocks_by_id
from app.rag.chunking.semantic.types import SectionNode, SourceBlock, estimate_tokens

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _render_block(block: SourceBlock) -> str:
    text = (block.text or "").strip()
    if not text:
        return ""
    if block.is_heading:
        marker = "#" * max(1, min(3, int(block.level or 1)))
        return f"{marker} {text}"
    return text


def _split_text(text: str, max_chars: int) -> List[str]:
    """Split oversized text on paragraph/sentence boundaries, never mid-word."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        sentences = [part.strip() for part in _SENTENCE_RE.split(paragraph) if part.strip()]
        if len(sentences) <= 1:
            sentences = [paragraph]
        buffer = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if buffer:
                    units.append(buffer)
                    buffer = ""
                current = ""
                for word in sentence.split():
                    if current and len(current) + 1 + len(word) > max_chars:
                        units.append(current)
                        current = word
                    else:
                        current = f"{current} {word}".strip()
                if current:
                    units.append(current)
            elif buffer and len(buffer) + 1 + len(sentence) > max_chars:
                units.append(buffer)
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            units.append(buffer)

    pieces: List[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > max_chars:
            pieces.append(current.strip())
            current = unit
        else:
            current = f"{current}\n\n{unit}".strip() if current else unit
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _table_rows(table: Dict[str, Any]) -> Tuple[List[str], List[List[str]], str]:
    headers = [str(cell or "") for cell in (table.get("headers") or [])]
    rows = [[str(cell or "") for cell in (row or [])] for row in (table.get("rows") or [])]
    caption = str(table.get("caption") or "")
    return headers, rows, caption


def _render_table_part(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    caption: str,
    context: str,
) -> str:
    """Render a table part as Markdown with repeated headers.

    Linearized facts are intentionally omitted for split parts: they can expand
    a wide table far beyond the embedding token budget. The complete grid is
    still preserved in ``table_json`` on every part.
    """
    width = max(
        [len(headers)] + [len(row) for row in rows] or [0]
    )
    parts: List[str] = []
    if context:
        parts.append(context)
    if caption:
        parts.append(caption)
    if width:
        header_cells = (list(headers) + [""] * width)[:width]
        lines = [
            "| " + " | ".join(str(cell or "").replace("|", "\\|") for cell in header_cells) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        for row in rows:
            cells = (list(row) + [""] * width)[:width]
            lines.append(
                "| " + " | ".join(str(cell or "").replace("|", "\\|") for cell in cells) + " |"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(part for part in parts if part)


def _split_table_block(
    block: SourceBlock,
    *,
    context: str,
    max_chars: int,
) -> List[Dict[str, Any]]:
    """Split a large table by rows, repeating headers and caption in each part."""
    table = dict(block.table or {})
    headers, rows, caption = _table_rows(table)
    if not rows:
        return [
            {
                "text": _render_block(block),
                "blocks": [block],
                "table_part": 1,
                "row_start": 0,
                "row_count": 0,
            }
        ]

    # Account for the context, caption and repeated Markdown header before
    # deciding how many rows fit in one part.
    width = max([len(headers)] + [len(row) for row in rows] or [0])
    header_chars = (
        sum(len(cell or "") for cell in headers)
        + 3 * max(1, width)
        + 8
    )
    separator_chars = 5 * max(1, width) + 5
    budget = max(
        120,
        max_chars
        - len(context)
        - len(caption)
        - header_chars
        - separator_chars
        - 8,
    )
    parts: List[Dict[str, Any]] = []
    current: List[List[str]] = []
    current_len = 0
    start = 0

    def emit(end: int) -> None:
        nonlocal current, current_len, start
        if not current:
            return
        parts.append(
            {
                "text": _render_table_part(
                    headers, current, caption=caption, context=context
                ),
                "blocks": [block],
                "table_part": len(parts) + 1,
                "row_start": start,
                "row_count": len(current),
            }
        )
        start = end
        current = []
        current_len = 0

    for index, row in enumerate(rows):
        row_len = sum(len(str(cell or "")) for cell in row) + 3 * len(row) + 2
        if current and current_len + row_len > budget:
            emit(index)
        current.append(row)
        current_len += row_len
    emit(len(rows))

    return parts or [
        {
            "text": _render_block(block),
            "blocks": [block],
            "table_part": 1,
            "row_start": 0,
            "row_count": len(rows),
        }
    ]


def _build_units(
    section_blocks: Sequence[SourceBlock],
    *,
    context: str,
    max_chars: int,
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for block in section_blocks:
        text = _render_block(block)
        if not text:
            continue
        if block.is_table and len(text) + len(context) + 2 > max_chars:
            units.extend(_split_table_block(block, context=context, max_chars=max_chars))
            continue
        if len(text) + len(context) + 2 > max_chars:
            for piece in _split_text(text, max(200, max_chars - len(context) - 2)):
                units.append({"text": piece, "blocks": [block], "table_part": None})
            continue
        units.append({"text": text, "blocks": [block], "table_part": None})
    return units


def _dominant_content_type(blocks: Sequence[SourceBlock]) -> str:
    if any(block.is_table for block in blocks):
        return "table"
    types = [block.content_type for block in blocks if not block.is_heading]
    if "list" in types:
        return "list"
    if "figure_caption" in types:
        return "figure_caption"
    if types:
        return types[0]
    return "heading"


def _complete_table_json(
    block: SourceBlock,
    *,
    table_part: Optional[int],
    row_start: Optional[int],
    row_count: Optional[int],
) -> Dict[str, Any]:
    table = dict(block.table or {})
    if table_part is not None:
        table["part"] = table_part
        table["part_row_start"] = row_start or 0
        table["part_row_count"] = row_count or 0
    return table


def generate_chunks(
    sections: Sequence[SectionNode],
    blocks: Sequence[SourceBlock],
    *,
    document_id: str = "",
    document_name: str = "",
    index_version: str = "",
    max_tokens: int = 1800,
) -> List[Dict[str, Any]]:
    """Emit chunk dicts whose text is reconstructed from the source blocks."""
    block_map = blocks_by_id(blocks)
    max_chars = max(200, int(max_tokens) * 4)

    table_index: Dict[str, int] = {}
    for block in blocks:
        if not block.is_table:
            continue
        table = block.table or {}
        table_id = str(table.get("table_id") or "")
        if table_id and table_id not in table_index:
            table_index[table_id] = len(table_index)

    chunks: List[Dict[str, Any]] = []

    for node in sections:
        own_blocks = [
            block_map[block_id]
            for block_id in node.block_ids
            if block_id in block_map
        ]
        own_blocks.sort(key=lambda block: block.order)
        if not own_blocks:
            continue

        context = " > ".join(part for part in node.path if part)
        if (
            len(node.path) == 1
            and own_blocks[0].is_heading
            and own_blocks[0].text.strip() == node.title
        ):
            # Avoid duplicating the heading as both context and first block.
            context = ""
        units = _build_units(own_blocks, context=context, max_chars=max_chars)
        if not units:
            continue

        group: List[Dict[str, Any]] = []
        group_len = len(context)

        def flush() -> None:
            nonlocal group, group_len
            if not group:
                return
            _emit_chunk(
                group,
                node=node,
                context=context,
                chunks=chunks,
                table_index=table_index,
                document_id=document_id,
                document_name=document_name,
                index_version=index_version,
            )
            group = []
            group_len = len(context)

        for unit in units:
            unit_len = len(unit["text"]) + 2
            if group and group_len + unit_len > max_chars:
                flush()
            group.append(unit)
            group_len += unit_len
        flush()

    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
        chunk["id"] = f"{document_id}_{index}" if document_id else f"chunk_{index}"
    return chunks


def _emit_chunk(
    units: Sequence[Dict[str, Any]],
    *,
    node: SectionNode,
    context: str,
    chunks: List[Dict[str, Any]],
    table_index: Dict[str, int],
    document_id: str,
    document_name: str,
    index_version: str,
) -> None:
    used_blocks: List[SourceBlock] = []
    seen: set[str] = set()
    for unit in units:
        for block in unit["blocks"]:
            if block.block_id not in seen:
                seen.add(block.block_id)
                used_blocks.append(block)
    if not used_blocks:
        return

    text = "\n\n".join(
        part for part in [context, *(unit["text"] for unit in units)] if part
    ).strip()
    if not text:
        return

    pages = [block.page_start for block in used_blocks] + [
        block.page_end for block in used_blocks
    ]
    page_start = min(pages) if pages else 1
    page_end = max(pages) if pages else page_start
    extraction_method = (
        "ocr" if any(block.extraction_method == "ocr" for block in used_blocks) else "native"
    )
    content_type = _dominant_content_type(used_blocks)
    table_blocks = [block for block in used_blocks if block.is_table]

    path = list(node.path)
    chunk: Dict[str, Any] = {
        "text": text,
        "page": page_start,
        "page_start": page_start,
        "page_end": page_end,
        "chunk_index": len(chunks),
        "chunk_type": "table" if table_blocks else "text",
        "content_type": content_type,
        "section": path[0] if path else node.title,
        "subsection": path[-1] if len(path) > 1 else "",
        "extraction_method": extraction_method,
        "section_id": node.section_id,
        "parent_section_id": node.parent_section_id,
        "section_level": node.level,
        "section_path": context,
        "source_block_ids": json.dumps([block.block_id for block in used_blocks]),
        "index_version": index_version,
        "layout_metadata": json.dumps(
            {
                "bboxes": [block.bbox for block in used_blocks if block.bbox],
                "heading_levels": [
                    block.level for block in used_blocks if block.is_heading
                ],
                "element_types": [block.element_type for block in used_blocks],
                "source_block_ids": [block.block_id for block in used_blocks],
            },
            ensure_ascii=False,
        ),
    }
    if document_id:
        chunk["document_id"] = str(document_id)
    if document_name:
        chunk["document_name"] = document_name

    if table_blocks:
        table_block = table_blocks[0]
        table = _complete_table_json(
            table_block,
            table_part=next(
                (unit.get("table_part") for unit in units if unit.get("table_part")),
                None,
            ),
            row_start=next(
                (unit.get("row_start") for unit in units if unit.get("row_start") is not None),
                None,
            ),
            row_count=next(
                (unit.get("row_count") for unit in units if unit.get("row_count") is not None),
                None,
            ),
        )
        table_id = str(table.get("table_id") or "")
        chunk["table_json"] = json.dumps(table, ensure_ascii=False)
        if table_id:
            chunk["table_id"] = table_id
            chunk["table_index"] = table_index.get(table_id, 0)
            chunk["table_ids"] = json.dumps([table_id])
        part = next(
            (unit.get("table_part") for unit in units if unit.get("table_part")), None
        )
        if part is not None:
            chunk["table_part"] = int(part)

    chunks.append(chunk)
