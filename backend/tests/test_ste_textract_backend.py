"""Unit tests for STE Textract AnalyzeDocument backend registry."""
from __future__ import annotations

from ste.backends.registry import (
    available_backends,
    create_backend,
    resolve_backend_name,
)


def test_resolve_backend_maps_legacy_docling_to_textract():
    assert resolve_backend_name("docling") == "textract"
    assert resolve_backend_name("docling_tableformer") == "textract"
    assert resolve_backend_name("textract") == "textract"
    assert resolve_backend_name("auto") in {"textract", "pymupdf"}


def test_create_textract_backend():
    backend = create_backend("textract", options={"dpi": 150})
    assert backend.name == "textract"
    assert backend.dpi == 150


def test_available_backends_excludes_docling():
    names = available_backends()
    assert "docling" not in names
