"""Configuration for Textract AnalyzeDocument extraction and optional Bedrock.

All settings are explicit dataclass fields with environment-variable
defaults, so runs are reproducible from a config dump alone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "off", "no"}


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    return default


def env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for one extraction run."""

    backend: str = "textract"  # textract | pymupdf | auto
    gate: str = "cues"  # cues (cheap, default) | layout (every page to backend)
    dpi: int = 150
    tables_only: bool = True
    pages: Tuple[int, ...] = ()
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    ocr_engine: str = "tesseract"  # used by pymupdf fallback only
    stitch_multipage: bool = True
    bedrock: bool = True
    bedrock_model_id: Optional[str] = None
    aws_region: Optional[str] = None
    max_workers: int = 4
    bedrock_max_workers: int = 3
    bedrock_skip_simple: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from ``STE_*`` environment variables."""
        backend = (os.environ.get("STE_BACKEND") or "textract").lower()
        # Legacy Docling backend name → Textract AnalyzeDocument.
        if backend in {"docling", "docling_tableformer"}:
            backend = "textract"
        gate = (os.environ.get("STE_GATE") or "cues").lower()
        ocr = (os.environ.get("STE_OCR") or "tesseract").lower()
        cache_dir = os.environ.get("STE_CACHE_DIR")
        return cls(
            backend=backend,
            gate=gate,
            dpi=env_int("STE_DPI", 150) or 150,
            tables_only=env_bool("STE_TABLES_ONLY", True),
            use_cache=env_bool("STE_CACHE", True),
            cache_dir=Path(cache_dir) if cache_dir else None,
            ocr_engine=ocr,
            stitch_multipage=env_bool("STE_STITCH_MULTIPAGE", True),
            bedrock=env_bool("STE_BEDROCK", True),
            bedrock_model_id=os.environ.get("STE_BEDROCK_MODEL_ID"),
            aws_region=os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION"),
            max_workers=env_int("STE_MAX_WORKERS", 4) or 4,
            bedrock_max_workers=env_int("STE_BEDROCK_MAX_WORKERS", 3) or 3,
            bedrock_skip_simple=env_bool("STE_BEDROCK_SKIP_SIMPLE", True),
        )

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """Return a copy with only non-``None`` overrides applied."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        # Drop removed Docling kwargs so older call sites keep working.
        for legacy in (
            "docling_table_mode",
            "docling_cell_matching",
            "docling_device",
            "docling_num_threads",
            "docling_artifacts_path",
        ):
            clean.pop(legacy, None)
        if clean.get("backend") in {"docling", "docling_tableformer"}:
            clean["backend"] = "textract"
        return replace(self, **clean)

    def backend_options(self) -> Dict[str, Any]:
        """Options passed to the selected extraction backend."""
        return {
            "ocr_engine": self.ocr_engine,
            "stitch_multipage": self.stitch_multipage,
            "dpi": self.dpi,
            "tables_only": self.tables_only,
            "aws_region": self.aws_region,
            "max_workers": self.max_workers,
        }

    def public_dict(self) -> Dict[str, Any]:
        """Serializable settings (used in run metadata)."""
        return {
            "backend": self.backend,
            "gate": self.gate,
            "dpi": self.dpi,
            "tables_only": self.tables_only,
            "pages": list(self.pages),
            "use_cache": self.use_cache,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "ocr_engine": self.ocr_engine,
            "stitch_multipage": self.stitch_multipage,
            "bedrock": self.bedrock,
            "bedrock_model_id": self.bedrock_model_id,
            "aws_region": self.aws_region,
            "max_workers": self.max_workers,
            "bedrock_max_workers": self.bedrock_max_workers,
            "bedrock_skip_simple": self.bedrock_skip_simple,
        }
