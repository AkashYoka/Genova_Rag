"""Unit tests for unified ingest stages + table extraction plan."""
from __future__ import annotations

import types

from app.rag.ingestion.pdf_layout.table_extraction import resolve_table_engine
from app.rag.ingestion.stages import (
    STAGE_LAYOUT_EXTRACT,
    STAGE_SEMANTIC_CHUNK,
    IngestContext,
    IngestPipeline,
)


def test_resolve_table_engine_prefers_ste():
    settings = types.SimpleNamespace(
        PDF_STE_TABLES_ENABLED=True,
        PDF_TEXTRACT_FALLBACK_ENABLED=True,
        PDF_TEXTRACT_TABLES_ENABLED=True,
    )
    assert resolve_table_engine(settings) == "ste_textract_analyze_bedrock"


def test_resolve_table_engine_textract_when_ste_off():
    settings = types.SimpleNamespace(
        PDF_STE_TABLES_ENABLED=False,
        PDF_TEXTRACT_FALLBACK_ENABLED=True,
        PDF_TEXTRACT_TABLES_ENABLED=True,
    )
    assert resolve_table_engine(settings) == "textract_detect"


def test_ingest_pipeline_records_stages_in_order():
    ctx = IngestContext(
        document_id="d1",
        user_id="u1",
        original_name="x.pdf",
        filepath="/tmp/x.pdf",
    )
    seen = []

    def on_progress(stage: str, pct: int) -> None:
        seen.append((stage, pct))

    pipeline = IngestPipeline(ctx, on_progress=on_progress)
    pipeline.run_stage(STAGE_LAYOUT_EXTRACT, lambda: None)
    pipeline.run_stage(STAGE_SEMANTIC_CHUNK, lambda: "ok")

    assert [s.name for s in ctx.stages] == [
        STAGE_LAYOUT_EXTRACT,
        STAGE_SEMANTIC_CHUNK,
    ]
    assert all(s.ok for s in ctx.stages)
    assert any(s == STAGE_LAYOUT_EXTRACT for s, _ in seen)
    assert any(s == STAGE_SEMANTIC_CHUNK for s, _ in seen)
    report = ctx.report()
    assert len(report["stages"]) == 2
