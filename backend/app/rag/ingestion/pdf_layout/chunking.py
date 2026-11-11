"""Semantic chunking over structured document elements.

Chunks follow headings, paragraphs, lists, tables, and captions instead of
arbitrary character windows. Oversized tables are split by rows with the
column header repeated in every part.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.config import get_settings
from app.rag.ingestion.pdf_layout.types import DocumentElement, normalize_ws

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _text_len(elements: Sequence[DocumentElement]) -> int:
    return sum(len(element.text) + 2 for element in elements)


def _split_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """Split on paragraph/sentence boundaries, never mid-word unless forced."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        sentences = [s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip()]
        if len(sentences) <= 1:
            sentences = [paragraph]
        buffer = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if buffer:
                    units.append(buffer)
                    buffer = ""
                words = sentence.split()
                current = ""
                for word in words:
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

    chunks: List[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > max_chars:
            chunks.append(current.strip())
            current = _overlap_tail(current, overlap) + "\n\n" + unit
        else:
            current = f"{current}\n\n{unit}".strip() if current else unit
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = text[-overlap:].strip()
    if not tail:
        return ""
    # Start at a word boundary.
    space = tail.find(" ")
    return tail[space + 1:] if space >= 0 else tail


def chunk_elements(
    elements: Sequence[DocumentElement],
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    settings = get_settings()
    chunk_size = int(chunk_size or settings.CHUNK_SIZE)
    chunk_overlap = int(chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP)
    max_table_chars = max(chunk_size, int(settings.PDF_TABLE_MAX_CHARS))

    chunks: List[Dict[str, Any]] = []
    buffer: List[DocumentElement] = []
    chunk_section = ""
    chunk_subsection = ""

    def add_chunk(
        text: str,
        source_elements: Sequence[DocumentElement],
        *,
        content_type: str,
        chunk_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        text = text.strip()
        if not text:
            return
        pages = [element.page_start for element in source_elements] or [1]
        page_end = max([element.page_end for element in source_elements] or pages)
        methods = {element.extraction_method for element in source_elements}
        extraction_method = "ocr" if "ocr" in methods else "native"
        meta = dict(metadata or {})
        layout_meta = {
            "bboxes": [
                list(element.bbox.as_tuple())
                for element in source_elements
                if element.bbox is not None
            ],
            "heading_levels": [
                element.level for element in source_elements if element.element_type == "heading"
            ],
            "element_types": [element.element_type for element in source_elements],
            "ocr_confidence": _mean_confidence(source_elements),
        }
        meta["layout_metadata"] = json.dumps(layout_meta, ensure_ascii=False)
        chunk = {
            "text": text,
            "page": min(pages),
            "page_start": min(pages),
            "page_end": page_end,
            "chunk_index": len(chunks),
            "chunk_type": chunk_type,
            "content_type": content_type,
            "section": source_elements[0].section if source_elements else chunk_section,
            "subsection": source_elements[0].subsection if source_elements else chunk_subsection,
            "extraction_method": extraction_method,
            **meta,
        }
        if document_id:
            chunk["document_id"] = str(document_id)
        if document_name:
            chunk["document_name"] = document_name
        chunks.append(chunk)

    def flush_buffer() -> None:
        nonlocal buffer, chunk_section, chunk_subsection
        if not buffer:
            return
        text = _render_buffer(buffer)
        for part in _split_text(text, chunk_size, chunk_overlap):
            add_chunk(part, buffer, content_type=_buffer_content_type(buffer))
        buffer = []
        chunk_section = ""
        chunk_subsection = ""

    for element in elements:
        if element.element_type == "heading":
            # Keep a heading attached to following content, but do not let a
            # stale buffer span unrelated sections.
            if buffer and _text_len(buffer) >= chunk_size * 0.6:
                flush_buffer()
            if not buffer:
                chunk_section = element.section
                chunk_subsection = element.subsection
            buffer.append(element)
            continue

        if element.element_type == "table":
            flush_buffer()
            chunk_section = element.section
            chunk_subsection = element.subsection
            table_chunks = _chunk_table(
                element,
                chunk_size=max_table_chars,
                document_id=document_id,
                document_name=document_name,
                chunk_index_start=len(chunks),
                section=element.section,
                subsection=element.subsection,
            )
            chunks.extend(table_chunks)
            continue

        if element.element_type in {"figure", "caption"}:
            if buffer and _text_len(buffer) + len(element.text) > chunk_size:
                flush_buffer()
            buffer.append(element)
            continue

        if buffer and _text_len(buffer) + len(element.text) > chunk_size:
            flush_buffer()
        if not buffer:
            chunk_section = element.section
            chunk_subsection = element.subsection
        buffer.append(element)

    flush_buffer()
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
    return chunks


def _render_buffer(buffer: Sequence[DocumentElement]) -> str:
    parts: List[str] = []
    for element in buffer:
        text = element.text.strip()
        if not text:
            continue
        if element.element_type == "heading":
            marker = "#" * max(1, min(3, element.level or 1))
            parts.append(f"{marker} {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _buffer_content_type(buffer: Sequence[DocumentElement]) -> str:
    types = [element.content_type for element in buffer if element.element_type != "heading"]
    if not types:
        return "heading"
    if "table" in types:
        return "table"
    if "figure_caption" in types:
        return "figure_caption"
    if "list" in types:
        return "list"
    return "paragraph"


def _mean_confidence(elements: Sequence[DocumentElement]) -> float:
    values: List[float] = []
    for element in elements:
        value = element.metadata.get("ocr_confidence")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(sum(values) / len(values), 2) if values else 0.0


def _chunk_table(
    element: DocumentElement,
    *,
    chunk_size: int,
    document_id: Optional[str],
    document_name: Optional[str],
    chunk_index_start: int,
    section: str,
    subsection: str,
) -> List[Dict[str, Any]]:
    table = dict(element.metadata.get("table") or {})
    headers: List[str] = list(table.get("headers") or [])
    rows: List[List[str]] = [list(row) for row in (table.get("rows") or [])]
    caption = table.get("caption") or element.metadata.get("caption") or ""
    from app.rag.tables.schema import make_table_id

    table_id = str(table.get("table_id") or "").strip() or make_table_id(
        document_id=document_id or "",
        page=element.page_start,
        bbox=table.get("bbox"),
        headers=headers,
        rows=rows,
    )
    # Every part keeps the COMPLETE table so retrieval can attach the whole
    # grid even when only one row fragment was hit.
    complete_table = {
        **table,
        "table_id": table_id,
        "headers": list(headers),
        "rows": [list(row) for row in rows],
    }
    header_text = _table_header_text(headers)
    prefix_parts = [part for part in (section, subsection) if part]
    context = "\n".join(prefix_parts)
    if caption:
        context = f"{context}\n{caption}".strip()
    base = "\n\n".join(part for part in (context, header_text) if part)
    if not rows:
        return [
            {
                "text": (base + "\n\n" + element.text).strip(),
                "page": element.page_start,
                "page_start": element.page_start,
                "page_end": element.page_end,
                "chunk_index": chunk_index_start,
                "chunk_type": "table",
                "content_type": "table",
                "section": section,
                "subsection": subsection,
                "extraction_method": element.extraction_method,
                "table_id": table_id,
                "table_json": json.dumps(complete_table, ensure_ascii=False),
                "layout_metadata": json.dumps({"bboxes": [list(element.bbox.as_tuple())] if element.bbox else []}),
                **({"document_id": str(document_id)} if document_id else {}),
                **({"document_name": document_name} if document_name else {}),
            }
        ]

    # Prefer section + caption + markdown + linearized facts for retrieval.
    from app.rag.ingestion.pdf_layout.table_normalize import render_table_with_facts

    rich_text = render_table_with_facts(
        headers,
        rows,
        caption=caption,
        section=section,
        subsection=subsection,
        include_markdown=True,
    )
    full_text = rich_text.strip() or "\n\n".join(part for part in (context, element.text) if part)
    if len(full_text) <= chunk_size:
        return [
            {
                "text": full_text,
                "page": element.page_start,
                "page_start": element.page_start,
                "page_end": element.page_end,
                "chunk_index": chunk_index_start,
                "chunk_type": "table",
                "content_type": "table",
                "section": section,
                "subsection": subsection,
                "extraction_method": element.extraction_method,
                "table_id": table_id,
                "table_json": json.dumps(complete_table, ensure_ascii=False),
                "layout_metadata": json.dumps({"bboxes": [list(element.bbox.as_tuple())] if element.bbox else []}),
                **({"document_id": str(document_id)} if document_id else {}),
                **({"document_name": document_name} if document_name else {}),
            }
        ]

    # 2) Split by rows, repeating headers and context in every part.
    chunks: List[Dict[str, Any]] = []
    available = max(200, chunk_size - len(base) - 2)
    current: List[List[str]] = []
    current_len = 0

    def emit() -> None:
        nonlocal current, current_len
        if not current:
            return
        part_table = {
            **complete_table,
            "part": len(chunks) + 1,
            "part_row_start": (
                table.get("rows", []).index(current[0])
                if current and current[0] in table.get("rows", [])
                else 0
            ),
            "part_row_count": len(current),
        }
        row_markdown = "\n".join(_table_row_markdown(headers, row) for row in current)
        part_facts = render_table_with_facts(
            headers,
            current,
            caption=caption,
            section="",
            subsection="",
            include_markdown=False,
        )
        text = "\n\n".join(
            part for part in (context, header_text, row_markdown, part_facts) if part
        )
        chunks.append(
            {
                "text": text,
                "page": element.page_start,
                "page_start": element.page_start,
                "page_end": element.page_end,
                "chunk_index": chunk_index_start + len(chunks),
                "chunk_type": "table",
                "content_type": "table",
                "section": section,
                "subsection": subsection,
                "extraction_method": element.extraction_method,
                "table_id": table_id,
                "table_json": json.dumps(part_table, ensure_ascii=False),
                "table_part": len(chunks) + 1,
                "layout_metadata": json.dumps(
                    {
                        "bboxes": [list(element.bbox.as_tuple())] if element.bbox else [],
                        "table_split": True,
                        "row_start": table.get("rows", []).index(current[0]) if current[0] in table.get("rows", []) else 0,
                    }
                ),
                **({"document_id": str(document_id)} if document_id else {}),
                **({"document_name": document_name} if document_name else {}),
            }
        )
        current = []
        current_len = 0

    for row in rows:
        row_len = len(_table_row_markdown(headers, row)) + 1
        if current and current_len + row_len > available:
            emit()
        current.append(row)
        current_len += row_len
    emit()
    return chunks


def _table_header_text(headers: Sequence[str]) -> str:
    if not headers:
        return ""
    width = len(headers)
    return "\n".join(
        [
            "| " + " | ".join(_cell(cell) for cell in headers) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
    )


def _table_row_markdown(headers: Sequence[str], row: Sequence[str]) -> str:
    width = max(len(headers), len(row))
    values = list(row) + [""] * (width - len(row))
    return "| " + " | ".join(_cell(cell) for cell in values) + " |"


def _cell(value: Any) -> str:
    return normalize_ws(str(value or "")).replace("|", "\\|")


def chunk_quality_report(chunks: Sequence[Dict[str, Any]], page_count: int) -> Dict[str, Any]:
    texts = [chunk.get("text", "") for chunk in chunks]
    hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts if text]
    lengths = [len(text) for text in texts if text]
    pages = {chunk.get("page_start", chunk.get("page")) for chunk in chunks}
    content_types: Dict[str, int] = {}
    for chunk in chunks:
        key = str(chunk.get("content_type", "paragraph"))
        content_types[key] = content_types.get(key, 0) + 1
    required = ("document_id", "page_start", "page_end", "section", "content_type", "extraction_method", "layout_metadata")
    complete = 0
    for chunk in chunks:
        if all(chunk.get(field) not in (None, "") for field in required):
            complete += 1
    return {
        "chunk_count": len(chunks),
        "pages_covered": len(pages),
        "page_count": page_count,
        "pages_missing": sorted(set(range(1, page_count + 1)) - {int(p) for p in pages if p}),
        "avg_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "duplicate_chunks": len(hashes) - len(set(hashes)),
        "content_types": content_types,
        "tables": sum(1 for chunk in chunks if chunk.get("chunk_type") == "table"),
        "ocr_chunks": sum(1 for chunk in chunks if chunk.get("extraction_method") == "ocr"),
        "metadata_complete": complete,
        "metadata_complete_ratio": round(complete / len(chunks), 4) if chunks else 0.0,
    }
