"""Docling-only mode: Textract fallback must never run when disabled."""
from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pymupdf as fitz
import pytest

from app.rag.ingestion.pdf_layout import parser as parser_mod
from app.rag.ingestion.pdf_layout.parser import (
    _table_engine,
    _textract_fallback_allowed,
)
from app.rag.ingestion.pdf_layout.types import BBox, TableData


def test_fallback_allowed_requires_both_flags():
    assert (
        _textract_fallback_allowed(
            SimpleNamespace(
                PDF_TEXTRACT_FALLBACK_ENABLED=False,
                PDF_TEXTRACT_TABLES_ENABLED=True,
            )
        )
        is False
    )
    assert (
        _textract_fallback_allowed(
            SimpleNamespace(
                PDF_TEXTRACT_FALLBACK_ENABLED=True,
                PDF_TEXTRACT_TABLES_ENABLED=False,
            )
        )
        is False
    )
    assert (
        _textract_fallback_allowed(
            SimpleNamespace(
                PDF_TEXTRACT_FALLBACK_ENABLED=True,
                PDF_TEXTRACT_TABLES_ENABLED=True,
            )
        )
        is True
    )


def test_table_engine_label_docling_only(monkeypatch):
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.parser._textract_fallback_allowed",
        lambda settings=None: False,
    )
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.parser.get_settings",
        lambda: SimpleNamespace(PDF_DOCLING_TABLES_ENABLED=True),
    )
    assert _table_engine() == "docling_only"


def test_table_engine_label_textract_primary(monkeypatch):
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.parser._textract_fallback_allowed",
        lambda settings=None: True,
    )
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.parser.get_settings",
        lambda: SimpleNamespace(PDF_DOCLING_TABLES_ENABLED=False),
    )
    assert _table_engine() == "textract_detect"


def test_extract_tables_from_page_skips_aws_when_fallback_disabled(monkeypatch):
    from app.rag.ingestion.pdf_layout import textract_tables as tt

    called = {"aws": False}

    def _boom(*_a, **_k):
        called["aws"] = True
        raise AssertionError("AWS Textract must not be called")

    monkeypatch.setattr(
        tt,
        "get_settings",
        lambda: SimpleNamespace(
            PDF_TEXTRACT_FALLBACK_ENABLED=False,
            PDF_TEXTRACT_TABLES_ENABLED=True,
            PDF_TEXTRACT_DPI=0,
            PDF_OCR_DPI=200,
            PDF_TEXTRACT_MIN_CONFIDENCE=50.0,
        ),
    )
    monkeypatch.setattr(tt, "_detect_document_text_bytes", _boom)
    monkeypatch.setattr(tt, "get_textract_client", _boom)
    page = MagicMock()
    assert tt.extract_tables_from_page(page, 1) == []
    assert called["aws"] is False


def test_quality_reject_marks_failed_without_keeping_rejected_grid():
    """Docling-only path keeps only accepted tables (rejected are not persisted)."""
    from app.rag.ingestion.pdf_layout.extraction_quality import filter_quality_tables

    bad = TableData(
        page=1,
        bbox=BBox(0, 0, 10, 10),
        headers=["Column 1", "Column 2"],
        rows=[["a", "b"]],
        extraction_engine="docling",
        structure_confidence=0.2,
    )
    good = TableData(
        page=1,
        bbox=BBox(0, 0, 10, 10),
        headers=["Year", "Revenue"],
        rows=[["2024", "10"]],
        extraction_engine="docling",
        structure_confidence=0.9,
    )
    accepted, rejected, _ = filter_quality_tables([bad, good])
    assert len(accepted) == 1
    assert accepted[0].headers == ["Year", "Revenue"]
    assert len(rejected) == 1
    final = list(accepted)
    assert all(t.extraction_engine == "docling" for t in final)
    assert not any(
        h.lower().startswith("column ") for t in final for h in (t.headers or [])
    )


def _docling_only_settings(**overrides):
    base = dict(
        PDF_LAYOUT_WRITE_DIAGNOSTICS=False,
        PDF_DOCLING_TABLES_ENABLED=True,
        PDF_DOCLING_TABLEFORMER_MODE="accurate",
        PDF_DOCLING_PAGE_BATCH_SIZE=4,
        PDF_TEXTRACT_FALLBACK_ENABLED=False,
        PDF_TEXTRACT_TABLES_ENABLED=True,
        PDF_TEXTRACT_TABULAR_ONLY=False,
        PDF_TEXTRACT_REGION="us-east-1",
        BEDROCK_REGION="us-east-1",
        PDF_OCR_ENABLED=False,
        PDF_OCR_DPI=150,
        PDF_CHUNK_SIZE=800,
        PDF_CHUNK_OVERLAP=100,
        PDF_TABLE_MAX_CHARS=4000,
        PDF_TABLE_SPLIT_ROWS=40,
        PDF_TABLE_QUALITY_MIN_COLS=2,
        PDF_TABLE_QUALITY_MIN_ROWS=1,
        PDF_TABLE_QUALITY_MAX_EMPTY_CELL_RATIO=0.65,
        PDF_TABLE_QUALITY_MIN_STRUCTURE_CONFIDENCE=0.35,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_docling_exception_does_not_invoke_textract_when_disabled(monkeypatch, tmp_path):
    """Docling raise → record failure; never call Textract in Docling-only mode."""
    pdf = tmp_path / "tiny.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Revenue table placeholder text for tabular page. " * 3)
    doc.save(str(pdf))
    doc.close()

    aws_called = {"n": 0}

    def _boom(*_a, **_k):
        raise RuntimeError("simulated docling OOM")

    def _textract_spy(*_a, **_k):
        aws_called["n"] += 1
        return []

    monkeypatch.setattr(parser_mod, "get_settings", lambda: _docling_only_settings())
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.docling_tables.extract_tables_from_pdf",
        _boom,
    )
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.textract_tables.extract_tables_from_page",
        _textract_spy,
    )
    monkeypatch.setattr(
        parser_mod, "_repair_page_tables", lambda tables, filepath="": tables
    )
    monkeypatch.setattr(parser_mod, "write_diagnostics", lambda _parsed: {})

    parsed = parser_mod.parse_pdf(str(pdf), document_name="tiny.pdf")
    assert aws_called["n"] == 0
    report = parsed.report["table_extraction"]
    assert report["engine"] == "docling_only"
    assert report["engine_info"]["textract_fallback_enabled"] is False
    page0 = parsed.pages[0]
    assert page0.metrics.get("docling_only_mode") is True or page0.metrics.get(
        "textract_skipped"
    ) == "disabled"
    assert page0.metrics.get("docling_extraction_failed_reason") or page0.metrics.get(
        "docling_error"
    )
    assert any("docling_only" in w for w in (page0.warnings or []))
