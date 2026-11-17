"""Named stages for the unified document ingestion pipeline.

One ordered flow owns progress + reporting:

  materialize → layout_extract → semantic_chunk → image_captions
  → table_rag → embed_index → hierarchy → enrichment → completed

PDF table engines (STE AnalyzeDocument + DetectDocumentText) run inside
``layout_extract`` via the table extraction coordinator — not as a sidecar.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Canonical stage ids (also stored on Document.processing_stage).
STAGE_MATERIALIZE = "materialize"
STAGE_LAYOUT_EXTRACT = "layout_extract"
STAGE_SEMANTIC_CHUNK = "semantic_chunk"
STAGE_IMAGE_CAPTIONS = "image_captions"
STAGE_TABLE_RAG = "table_rag"
STAGE_EMBED_INDEX = "embed_index"
STAGE_HIERARCHY = "hierarchy"
STAGE_ENRICHMENT = "enrichment"
STAGE_COMPLETED = "completed"
STAGE_FAILED = "failed"

# Overall progress bands so nested callbacks stay monotonic.
STAGE_PROGRESS = {
    STAGE_MATERIALIZE: (5, 12),
    STAGE_LAYOUT_EXTRACT: (12, 40),
    STAGE_SEMANTIC_CHUNK: (40, 55),
    STAGE_IMAGE_CAPTIONS: (55, 58),
    STAGE_TABLE_RAG: (58, 68),
    STAGE_EMBED_INDEX: (68, 90),
    STAGE_HIERARCHY: (90, 93),
    STAGE_ENRICHMENT: (93, 99),
    STAGE_COMPLETED: (100, 100),
}


ProgressFn = Callable[[str, int], None]


@dataclass
class StageRecord:
    name: str
    started_at: float
    ended_at: Optional[float] = None
    ok: bool = True
    detail: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def elapsed_s(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return round(self.ended_at - self.started_at, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "elapsed_s": self.elapsed_s,
            "error": self.error or None,
            "detail": dict(self.detail or {}),
        }


@dataclass
class IngestContext:
    """Shared state passed through every ingestion stage."""

    document_id: str
    user_id: str
    original_name: str
    filepath: str
    cleanup_path: Optional[str] = None
    page_count: int = 0
    requested_strategy: str = "semantic_llm"
    # Layout
    parsed: Any = None  # ParsedDocument when PDF layout_extract ran
    layout_report: Dict[str, Any] = field(default_factory=dict)
    # Chunking
    chunks: Any = None  # SemanticChunkList or list
    chunk_hierarchy: List[Any] = field(default_factory=list)
    chunk_report: Dict[str, Any] = field(default_factory=dict)
    chunk_strategy: str = ""
    chunk_index_version: str = ""
    # TableRAG
    store_chunk_list: List[Dict[str, Any]] = field(default_factory=list)
    table_records: List[Any] = field(default_factory=list)
    table_index_version: str = ""
    indexed_chunk_count: int = 0
    document_chunk_count: int = 0
    # Pipeline telemetry
    stages: List[StageRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def report(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "original_name": self.original_name,
            "page_count": self.page_count,
            "chunk_strategy": self.chunk_strategy,
            "index_version": self.chunk_index_version,
            "document_chunk_count": self.document_chunk_count,
            "indexed_chunk_count": self.indexed_chunk_count,
            "table_record_count": len(self.table_records or []),
            "layout_engine": (
                (self.layout_report.get("table_extraction") or {}).get("engine")
                or (self.chunk_report.get("ste_tables") or {}).get("extraction_engine")
            ),
            "stages": [s.to_dict() for s in self.stages],
            "errors": list(self.errors),
        }


class IngestPipeline:
    """Runs named stages with unified progress + stage records."""

    def __init__(
        self,
        ctx: IngestContext,
        *,
        on_progress: Optional[ProgressFn] = None,
    ) -> None:
        self.ctx = ctx
        self.on_progress = on_progress

    def progress(self, stage: str, pct: int) -> None:
        lo, hi = STAGE_PROGRESS.get(stage, (0, 100))
        mapped = max(lo, min(hi, int(pct)))
        if self.on_progress:
            try:
                self.on_progress(stage, mapped)
            except Exception:
                pass

    def map_nested(self, stage: str, local_pct: int) -> int:
        """Map a 0–100 nested callback into the stage progress band."""
        lo, hi = STAGE_PROGRESS.get(stage, (0, 100))
        local = max(0, min(100, int(local_pct)))
        return int(lo + (hi - lo) * (local / 100.0))

    def run_stage(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        start_pct: Optional[int] = None,
        end_pct: Optional[int] = None,
        detail: Optional[Dict[str, Any]] = None,
        fatal: bool = True,
    ) -> Any:
        lo, hi = STAGE_PROGRESS.get(name, (0, 100))
        self.progress(name, start_pct if start_pct is not None else lo)
        record = StageRecord(name=name, started_at=time.perf_counter())
        logger.info(
            "INGEST_STAGE start document_id=%s stage=%s",
            self.ctx.document_id,
            name,
        )
        try:
            result = fn()
            record.ok = True
            if detail:
                record.detail.update(detail)
            return result
        except Exception as exc:
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            self.ctx.errors.append(f"{name}: {record.error}")
            logger.exception(
                "INGEST_STAGE failed document_id=%s stage=%s: %s",
                self.ctx.document_id,
                name,
                exc,
            )
            if fatal:
                raise
            return None
        finally:
            record.ended_at = time.perf_counter()
            self.ctx.stages.append(record)
            self.progress(name, end_pct if end_pct is not None else hi)
            logger.info(
                "INGEST_STAGE end document_id=%s stage=%s ok=%s elapsed=%ss",
                self.ctx.document_id,
                name,
                record.ok,
                record.elapsed_s,
            )
