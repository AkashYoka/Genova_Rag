"""Shared typed structures for hierarchical semantic chunking."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def estimate_tokens(text: str) -> int:
    """Conservative token estimate without a tokenizer dependency.

    Titan Text Embeddings V2 and Nova both accept far more tokens than the
    English ~4 characters/token heuristic assumes, so rounding up keeps chunk
    sizes safely below the configured embedding limit.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass
class SourceBlock:
    """One immutable unit of extracted source content."""

    block_id: str
    order: int
    element_type: str  # heading | paragraph | list | table | figure | caption
    text: str
    page_start: int = 1
    page_end: int = 1
    level: int = 0
    section: str = ""
    subsection: str = ""
    extraction_method: str = "native"
    content_type: str = "paragraph"
    bbox: Optional[List[float]] = None
    table: Optional[Dict[str, Any]] = None

    @property
    def is_heading(self) -> bool:
        return self.element_type == "heading"

    @property
    def is_table(self) -> bool:
        return self.element_type == "table"

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)

    def preview(self, limit: int) -> str:
        text = " ".join((self.text or "").split())
        if limit > 0 and len(text) > limit:
            return text[: limit - 1].rstrip() + "…"
        return text


@dataclass
class SectionSpec:
    """Section boundary proposed by the LLM or the deterministic fallback."""

    title: str
    level: int
    block_ids: List[str] = field(default_factory=list)
    summary: str = ""
    ref: str = ""
    parent_ref: Optional[str] = None
    source: str = "llm"


@dataclass
class SectionNode:
    """Resolved node in the persisted section tree."""

    section_id: str
    title: str
    level: int
    parent_section_id: Optional[str]
    ordinal: int
    block_ids: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    summary: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    children: List["SectionNode"] = field(default_factory=list)

    def to_dict(self, *, include_children: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "section_id": self.section_id,
            "title": self.title,
            "level": self.level,
            "parent_section_id": self.parent_section_id,
            "ordinal": self.ordinal,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "summary": self.summary,
            "path": list(self.path),
            "block_ids": list(self.block_ids),
            "tables": list(self.tables),
        }
        if include_children:
            payload["children"] = [child.to_dict() for child in self.children]
        return payload


@dataclass
class ValidationReport:
    """Diagnostics proving no source content was silently dropped."""

    block_count: int = 0
    section_count: int = 0
    chunk_count: int = 0
    page_count: int = 0
    pages_covered: int = 0
    pages_missing: List[int] = field(default_factory=list)
    duplicate_block_ids: List[str] = field(default_factory=list)
    duplicate_count: int = 0
    unknown_block_ids: int = 0
    missing_block_count: int = 0
    repaired_block_count: int = 0
    source_coverage: float = 0.0
    token_limit_violations: int = 0
    chunk_types: Dict[str, int] = field(default_factory=dict)
    used_llm: bool = False
    used_fallback: bool = False
    llm_windows: int = 0
    llm_calls: int = 0
    llm_retries: int = 0
    strategy: str = ""
    index_version: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_count": self.block_count,
            "section_count": self.section_count,
            "chunk_count": self.chunk_count,
            "page_count": self.page_count,
            "pages_covered": self.pages_covered,
            "pages_missing": list(self.pages_missing),
            "duplicate_block_ids": list(self.duplicate_block_ids),
            "duplicate_count": self.duplicate_count,
            "unknown_block_ids": self.unknown_block_ids,
            "missing_block_count": self.missing_block_count,
            "repaired_block_count": self.repaired_block_count,
            "source_coverage": round(self.source_coverage, 4),
            "token_limit_violations": self.token_limit_violations,
            "chunk_types": dict(self.chunk_types),
            "used_llm": self.used_llm,
            "used_fallback": self.used_fallback,
            "llm_windows": self.llm_windows,
            "llm_calls": self.llm_calls,
            "llm_retries": self.llm_retries,
            "strategy": self.strategy,
            "index_version": self.index_version,
            "errors": list(self.errors),
        }


@dataclass
class AnalysisResult:
    """Result of LLM (or deterministic) section analysis."""

    sections: List[SectionSpec] = field(default_factory=list)
    used_llm: bool = False
    used_fallback: bool = False
    windows: int = 0
    llm_calls: int = 0
    retries: int = 0
    errors: List[str] = field(default_factory=list)
    raw_section_count: int = 0


class SemanticChunkList(list):
    """List-compatible chunk collection carrying hierarchy + diagnostics.

    Existing callers iterate the result as a list of chunk dicts. Ingestion
    reads the extra attributes to persist the section tree without changing the
    public ``chunk_document`` contract.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        *,
        hierarchy: Optional[List[SectionNode]] = None,
        report: Optional[Dict[str, Any]] = None,
        strategy: str = "semantic_llm",
        index_version: str = "",
        used_fallback: bool = False,
        page_count: int = 0,
        source_blocks: Optional[List[SourceBlock]] = None,
    ):
        super().__init__(chunks)
        self.hierarchy = hierarchy or []
        self.report = report or {}
        self.strategy = strategy
        self.index_version = index_version
        self.used_fallback = used_fallback
        self.page_count = page_count
        # Retained for diagnostics/audit (e.g. old-vs-new comparison scripts).
        self.source_blocks = source_blocks or []

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        return list(self)

    def hierarchy_dict(self) -> List[Dict[str, Any]]:
        return [node.to_dict() for node in self.hierarchy]

    def report_json(self) -> str:
        return json.dumps(self.report, ensure_ascii=False)
