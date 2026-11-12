"""Resolve validated section specs into a deterministic persisted tree."""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Optional, Sequence

from app.rag.chunking.semantic.blocks import blocks_by_id, page_span
from app.rag.chunking.semantic.types import SectionNode, SectionSpec, SourceBlock

_WS_RE = re.compile(r"\s+")
_MAX_TITLE = 500
_TOC_TITLE_RE = re.compile(
    r"(?i)^(?:table\s+of\s+)?contents\*?\s*$|prescribing\s+information:\s*contents"
)


def _is_toc_title(value: str) -> bool:
    return bool(_TOC_TITLE_RE.search(_WS_RE.sub(" ", str(value or "")).strip()))


def _clean_title(value: str, fallback: str) -> str:
    text = _WS_RE.sub(" ", str(value or "")).strip().lstrip("#").strip()
    if not text:
        text = fallback
    return text[:_MAX_TITLE]


def _section_id(document_id: str, path: Sequence[str], ordinal: int, used: set[str]) -> str:
    """Deterministic id from document id + hierarchy path.

    A duplicate sibling title would otherwise collide; the ordinal is only
    mixed in when the path alone has already been used.
    """
    base = "|".join([str(document_id or ""), *path])
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:20]
    candidate = f"sec_{digest}"
    if candidate in used:
        digest = hashlib.sha256(f"{base}|{ordinal}".encode("utf-8")).hexdigest()[:20]
        candidate = f"sec_{digest}"
    used.add(candidate)
    return candidate


def _section_tables(blocks: Iterable[SourceBlock]) -> List[Dict[str, object]]:
    tables: List[Dict[str, object]] = []
    for block in blocks:
        if not block.is_table:
            continue
        table = dict(block.table or {})
        table_id = str(table.get("table_id") or "").strip()
        tables.append(
            {
                "table_id": table_id,
                "page": block.page_start,
                "page_end": block.page_end,
                "caption": str(table.get("caption") or ""),
                "headers": list(table.get("headers") or []),
                "n_rows": int(table.get("n_rows") or len(table.get("rows") or [])),
                "n_cols": int(table.get("n_cols") or len(table.get("headers") or [])),
                "block_id": block.block_id,
            }
        )
    return tables


def _fallback_summary(blocks: Sequence[SourceBlock], title: str) -> str:
    for block in blocks:
        if block.is_heading:
            continue
        text = _WS_RE.sub(" ", block.text or "").strip()
        if text:
            return text[:240]
    return title


def build_hierarchy(
    specs: Sequence[SectionSpec],
    blocks: Sequence[SourceBlock],
    *,
    document_id: str = "",
    document_name: str = "",
    index_version: str = "",
) -> List[SectionNode]:
    """Build the section tree, honoring LLM parent refs then level nesting."""
    if not specs:
        return []

    by_id = blocks_by_id(blocks)
    ref_to_index = {spec.ref: index for index, spec in enumerate(specs) if spec.ref}
    used_ids: set[str] = set()
    nodes: List[SectionNode] = []
    stack: List[int] = []

    for index, spec in enumerate(specs):
        level = max(1, min(6, int(spec.level or 1)))
        parent_index: Optional[int] = None

        ref = spec.parent_ref
        if ref and ref in ref_to_index:
            candidate = ref_to_index[ref]
            if candidate < index and specs[candidate].level < level:
                # Never nest body chapters under a CONTENTS listing.
                if not _is_toc_title(specs[candidate].title):
                    parent_index = candidate

        while stack and specs[stack[-1]].level >= level:
            stack.pop()
        # Skip TOC ancestors left on the stack from weaker LLM nesting.
        while stack and _is_toc_title(specs[stack[-1]].title):
            stack.pop()
        if parent_index is None and stack:
            parent_index = stack[-1]
        if parent_index is not None and _is_toc_title(specs[parent_index].title):
            parent_index = None

        parent = nodes[parent_index] if parent_index is not None else None
        title = _clean_title(spec.title, f"Section {index + 1}")
        if _is_toc_title(title):
            title = "Table of Contents"
        path = (list(parent.path) if parent else []) + [title]
        section_blocks = [
            by_id[block_id]
            for block_id in spec.block_ids
            if block_id in by_id
        ]
        section_blocks.sort(key=lambda block: block.order)
        page_start, page_end = page_span(section_blocks)

        node = SectionNode(
            section_id=_section_id(document_id, path, index, used_ids),
            title=path[-1],
            level=level,
            parent_section_id=parent.section_id if parent else None,
            ordinal=index,
            block_ids=[block.block_id for block in section_blocks],
            page_start=page_start,
            page_end=page_end,
            summary=(spec.summary or _fallback_summary(section_blocks, path[-1]))[:500],
            tables=_section_tables(section_blocks),
            path=path,
        )
        nodes.append(node)
        if parent is not None:
            parent.children.append(node)
        # CONTENTS is a leaf listing — do not keep it open as a parent.
        if _is_toc_title(node.title):
            continue
        stack.append(index)

    return nodes


def flatten_sections(nodes: Sequence[SectionNode]) -> List[SectionNode]:
    """Return every node once, accepting either a flat list or tree roots."""
    flat: List[SectionNode] = []
    seen: set[str] = set()

    def visit(node: SectionNode) -> None:
        if node.section_id in seen:
            return
        seen.add(node.section_id)
        flat.append(node)
        for child in node.children:
            visit(child)

    for node in nodes:
        visit(node)
    return flat


def to_tree(nodes: Sequence[SectionNode]) -> List[SectionNode]:
    """Return only root nodes; children are already linked in place."""
    return [node for node in nodes if node.parent_section_id is None]


def root_sections(nodes: Sequence[SectionNode]) -> List[SectionNode]:
    return to_tree(nodes)


def hierarchy_summary(nodes: Sequence[SectionNode]) -> Dict[str, object]:
    flat = flatten_sections(nodes)
    return {
        "sections": len(flat),
        "root_sections": len(root_sections(nodes)),
        "max_depth": max((node.level for node in flat), default=0),
        "tables": sum(len(node.tables) for node in flat),
        "section_ids": [node.section_id for node in flat][:50],
    }
