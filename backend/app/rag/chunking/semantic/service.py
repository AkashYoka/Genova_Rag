"""Orchestrate layout extraction → section analysis → chunks → diagnostics."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from app.config import get_settings
from app.rag.chunking.semantic.analyzer import (
    analyze_blocks,
    deterministic_sections,
    validate_and_repair,
)
from app.rag.chunking.semantic.blocks import extract_source_blocks
from app.rag.chunking.semantic.generator import generate_chunks
from app.rag.chunking.semantic.hierarchy import build_hierarchy, flatten_sections
from app.rag.chunking.semantic.types import (
    SemanticChunkList,
    ValidationReport,
    estimate_tokens,
)
from app.rag.chunking.semantic.version import resolve_index_version, strategy_label

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int], None]


def should_use_semantic_chunking(
    filepath: str, *, strategy: Optional[str] = None
) -> bool:
    """Return True when this file should use the hierarchy pipeline.

    Fake paths used by tests and unsupported extensions stay on the legacy
    path, preserving backward compatibility for existing callers. ``strategy``
    lets a reindex request override the global ``CHUNK_STRATEGY`` per document.
    """
    strategy = (strategy or strategy_label()).lower()
    if strategy == "legacy":
        return False
    ext = str(filepath or "").rsplit(".", 1)[-1].lower()
    return ext == "pdf" and os.path.exists(filepath)


def _report_progress(progress: Optional[ProgressCallback], stage: str, value: int) -> None:
    if not progress:
        return
    try:
        progress(stage, int(value))
    except Exception:  # progress must never break ingestion
        logger.debug("Semantic chunking progress callback failed", exc_info=True)


def _pages_covered(chunks: List[Dict[str, Any]]) -> set[int]:
    pages: set[int] = set()
    for chunk in chunks:
        start = chunk.get("page_start") or chunk.get("page")
        end = chunk.get("page_end") or start
        try:
            start_i = int(start or 1)
            end_i = int(end or start_i)
        except Exception:
            continue
        for page in range(start_i, end_i + 1):
            pages.add(page)
    return pages


def _legacy_result(
    parsed: Any,
    *,
    strategy: str,
    reason: str,
) -> SemanticChunkList:
    chunks = list(getattr(parsed, "chunks", None) or [])
    page_count = int(getattr(parsed, "page_count", 0) or 0)
    for index, chunk in enumerate(chunks):
        chunk.setdefault("chunk_index", index)
    report = ValidationReport(
        block_count=0,
        section_count=0,
        chunk_count=len(chunks),
        page_count=page_count,
        pages_covered=len(_pages_covered(chunks)),
        used_fallback=True,
        strategy=strategy,
        index_version=resolve_index_version(strategy, used_fallback=True),
        errors=[reason],
    )
    return SemanticChunkList(
        chunks,
        hierarchy=[],
        report=report.to_dict(),
        strategy=strategy,
        index_version=report.index_version,
        used_fallback=True,
        page_count=page_count,
    )


def semantic_chunk_document(
    filepath: str,
    *,
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
    strategy: Optional[str] = None,
) -> SemanticChunkList:
    """Run hierarchical semantic chunking for a PDF on disk.

    ``strategy`` overrides the global setting for per-document reindexing.
    """
    settings = get_settings()
    strategy = (strategy or strategy_label()).lower()
    started = time.perf_counter()
    document_id = str(document_id or "")
    document_name = document_name or os.path.basename(filepath)

    _report_progress(progress, "extracting", 25)
    from app.rag.ingestion.pdf_layout.parser import parse_pdf

    parsed = parse_pdf(
        filepath,
        document_id=document_id or None,
        document_name=document_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    page_count = int(getattr(parsed, "page_count", 0) or 0)

    blocks = extract_source_blocks(getattr(parsed, "elements", None) or [])
    if not blocks:
        logger.warning(
            "Semantic chunking found no source blocks for %s; using layout chunks",
            filepath,
        )
        return _legacy_result(
            parsed,
            strategy=strategy,
            reason="no_source_blocks",
        )

    _report_progress(progress, "analyzing", 35)
    analysis = analyze_blocks(
        blocks,
        document_name=document_name,
        page_count=page_count,
        settings=settings,
        progress=progress,
        strategy=strategy,
    )

    specs, stats = validate_and_repair(analysis.sections, blocks)
    if not specs:
        specs = deterministic_sections(blocks, document_name=document_name)
        stats = {
            "duplicate_block_ids": [],
            "duplicate_count": 0,
            "unknown_block_ids": 0,
            "missing_block_count": len(blocks),
            "repaired_block_count": len(blocks),
            "source_coverage": 0.0,
        }
        analysis.used_fallback = True

    index_version = resolve_index_version(
        strategy, used_fallback=analysis.used_fallback
    )
    nodes = build_hierarchy(
        specs,
        blocks,
        document_id=document_id,
        document_name=document_name,
        index_version=index_version,
    )

    _report_progress(progress, "chunking", 55)
    max_tokens = int(
        getattr(settings, "SEMANTIC_CHUNKING_MAX_CHUNK_TOKENS", 1800) or 1800
    )
    chunks = generate_chunks(
        nodes,
        blocks,
        document_id=document_id,
        document_name=document_name,
        index_version=index_version,
        max_tokens=max_tokens,
    )

    if not chunks:
        # Deterministic headings may still produce no chunks for pathological
        # documents; fall back to the proven layout chunker.
        logger.warning(
            "Semantic chunking produced no chunks for %s; using layout chunks",
            filepath,
        )
        return _legacy_result(
            parsed,
            strategy=strategy,
            reason="no_semantic_chunks",
        )

    for chunk in chunks:
        chunk["token_count"] = estimate_tokens(chunk.get("text") or "")
        chunk["index_version"] = index_version

    covered = _pages_covered(chunks)
    missing_pages = [
        page for page in range(1, page_count + 1) if page not in covered
    ]
    token_violations = sum(
        1 for chunk in chunks if int(chunk.get("token_count") or 0) > max_tokens
    )
    chunk_types: Dict[str, int] = {}
    for chunk in chunks:
        key = str(chunk.get("chunk_type") or "text")
        chunk_types[key] = chunk_types.get(key, 0) + 1

    report = ValidationReport(
        block_count=len(blocks),
        section_count=len(flatten_sections(nodes)),
        chunk_count=len(chunks),
        page_count=page_count,
        pages_covered=len(covered),
        pages_missing=missing_pages,
        duplicate_block_ids=stats.get("duplicate_block_ids", []),
        duplicate_count=int(stats.get("duplicate_count", 0)),
        unknown_block_ids=int(stats.get("unknown_block_ids", 0)),
        missing_block_count=int(stats.get("missing_block_count", 0)),
        repaired_block_count=int(stats.get("repaired_block_count", 0)),
        source_coverage=float(stats.get("source_coverage", 0.0)),
        token_limit_violations=token_violations,
        chunk_types=chunk_types,
        used_llm=analysis.used_llm,
        used_fallback=analysis.used_fallback,
        llm_windows=analysis.windows,
        llm_calls=analysis.llm_calls,
        llm_retries=analysis.retries,
        strategy=strategy,
        index_version=index_version,
        errors=list(analysis.errors),
    )
    report_dict = report.to_dict()
    report_dict["elapsed_s"] = round(time.perf_counter() - started, 2)
    report_dict["chunk_size_hint"] = chunk_size
    report_dict["chunk_overlap_hint"] = chunk_overlap

    _report_progress(progress, "chunked", 60)
    logger.info(
        "Semantic chunking for %s: %d blocks → %d sections → %d chunks "
        "(strategy=%s version=%s fallback=%s)",
        filepath,
        len(blocks),
        report.section_count,
        len(chunks),
        strategy,
        index_version,
        analysis.used_fallback,
    )

    return SemanticChunkList(
        chunks,
        hierarchy=nodes,
        report=report_dict,
        strategy=strategy,
        index_version=index_version,
        used_fallback=analysis.used_fallback,
        page_count=page_count,
        source_blocks=list(blocks),
    )
