"""Unit tests for STE ingest performance flags (skip Detect, skip-simple Bedrock)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.rag.ingestion.pdf_layout.table_extraction import run_document_table_extraction
from app.rag.ingestion.pdf_layout.types import BBox, TableData
from ste.bedrock_reconstructor import should_skip_bedrock_simple
from ste.models import BBox as SteBBox
from ste.models import ExtractedTable


def _ste_table(page: int = 1) -> TableData:
    return TableData(
        page=page,
        bbox=BBox(10, 10, 200, 100),
        headers=["A", "B"],
        rows=[["1", "2"]],
        extraction_method="ste_textract_analyze_bedrock",
        extraction_engine="ste_textract_analyze_bedrock",
    )


def test_should_skip_bedrock_simple_high_confidence():
    table = ExtractedTable(
        table_id="t1",
        page=1,
        bbox=SteBBox(0, 0, 100, 100),
        headers=["A", "B", "C"],
        rows=[["1", "2", "3"], ["4", "5", "6"]],
        confidence=0.95,
    )
    skip, reason = should_skip_bedrock_simple(table)
    assert skip is True
    assert reason == "simple_high_confidence"


def test_should_not_skip_bedrock_sparse_or_low_conf():
    sparse = ExtractedTable(
        table_id="t2",
        page=1,
        bbox=SteBBox(0, 0, 100, 100),
        headers=["A", "B"],
        rows=[["1", ""], ["", ""]],
        confidence=0.99,
    )
    skip, reason = should_skip_bedrock_simple(sparse)
    assert skip is False
    assert reason == "sparse_grid"

    low = ExtractedTable(
        table_id="t3",
        page=1,
        bbox=SteBBox(0, 0, 100, 100),
        headers=["A", "B"],
        rows=[["1", "2"]],
        confidence=0.5,
    )
    skip2, reason2 = should_skip_bedrock_simple(low)
    assert skip2 is False
    assert reason2 == "low_confidence"


def test_skip_detect_when_ste_primary():
    settings = SimpleNamespace(
        PDF_STE_TABLES_ENABLED=True,
        PDF_STE_BEDROCK_ENABLED=True,
        PDF_STE_SKIP_DETECT_WHEN_STE=True,
        PDF_TEXTRACT_FALLBACK_ENABLED=True,
        PDF_TEXTRACT_TABLES_ENABLED=True,
        PDF_TEXTRACT_TABULAR_ONLY=True,
        PDF_TEXTRACT_REGION="us-east-1",
        BEDROCK_REGION="us-east-1",
    )
    ste_tables = [_ste_table()]
    ste_meta = {"ok": True, "elapsed_s": 1.2, "table_count": 1}

    with patch(
        "app.rag.ingestion.pdf_layout.ste_tables.extract_ste_tables_from_pdf",
        return_value=(ste_tables, ste_meta),
    ), patch(
        "app.rag.ingestion.pdf_layout.textract_tables.extract_tables_from_pdf_detect"
    ) as detect_mock:
        plan = run_document_table_extraction(
            "dummy.pdf",
            settings=settings,
            document_id="doc",
            document_name="dummy.pdf",
        )

    detect_mock.assert_not_called()
    assert plan.engine_info.get("detect_skipped") is True
    assert plan.textract_enabled is False
    assert len(plan.ste_tables) == 1
    assert plan.to_report()["detect_skipped"] is True


def test_detect_runs_when_skip_flag_false():
    settings = SimpleNamespace(
        PDF_STE_TABLES_ENABLED=True,
        PDF_STE_BEDROCK_ENABLED=True,
        PDF_STE_SKIP_DETECT_WHEN_STE=False,
        PDF_TEXTRACT_FALLBACK_ENABLED=True,
        PDF_TEXTRACT_TABLES_ENABLED=True,
        PDF_TEXTRACT_TABULAR_ONLY=True,
        PDF_TEXTRACT_REGION="us-east-1",
        BEDROCK_REGION="us-east-1",
    )
    ste_tables = [_ste_table()]
    ste_meta = {"ok": True, "elapsed_s": 1.0, "table_count": 1}

    with patch(
        "app.rag.ingestion.pdf_layout.ste_tables.extract_ste_tables_from_pdf",
        return_value=(ste_tables, ste_meta),
    ), patch(
        "app.rag.ingestion.pdf_layout.textract_tables.extract_tables_from_pdf_detect",
        return_value=([], {"pages_processed": 0, "words_by_page": {}}),
    ) as detect_mock, patch(
        "app.rag.ingestion.pdf_layout.textract_tables.get_textract_api_usage",
        return_value={},
    ):
        plan = run_document_table_extraction(
            "dummy.pdf",
            settings=settings,
        )

    detect_mock.assert_called_once()
    assert plan.engine_info.get("detect_skipped") is False


def test_textract_backend_accepts_max_workers():
    from ste.backends.registry import create_backend

    backend = create_backend(
        "textract", options={"dpi": 150, "max_workers": 6}
    )
    assert backend.max_workers == 6
    assert backend.dpi == 150
