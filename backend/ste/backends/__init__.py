"""Modular extraction backends (Textract AnalyzeDocument primary)."""
from ste.backends.base import DocumentExtraction, RawTable, TableExtractionBackend
from ste.backends.registry import (
    PymupdfBackend,
    TextractAnalyzeBackend,
    available_backends,
    create_backend,
    resolve_backend_name,
)

__all__ = [
    "DocumentExtraction",
    "RawTable",
    "TableExtractionBackend",
    "TextractAnalyzeBackend",
    "PymupdfBackend",
    "available_backends",
    "create_backend",
    "resolve_backend_name",
]
