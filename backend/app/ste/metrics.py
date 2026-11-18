"""Local processing metrics: wall/CPU time, peak RAM, hardware, cost."""
from __future__ import annotations

import os
import platform
import sys
import time
import tracemalloc
from typing import Any, Dict, Optional


def _rss_mb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def hardware_info() -> Dict[str, Any]:
    """Best-effort local hardware/software description for the run report."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        info["ram_total_gb"] = round(
            psutil.virtual_memory().total / (1024.0**3), 2
        )
    except Exception:
        info["ram_total_gb"] = None
    try:
        from importlib.metadata import version

        info["boto3_version"] = version("boto3")
    except Exception:
        info["boto3_version"] = None
    return info


class ResourceMonitor:
    """Measure wall time, CPU time and peak memory for one run."""

    def __init__(self) -> None:
        self._wall_start = 0.0
        self._cpu_start = 0.0
        self._rss_start: Optional[float] = None
        self._peak_rss: Optional[float] = None
        self._tracemalloc = False
        self._stopped = False

    def start(self) -> "ResourceMonitor":
        if not self._tracemalloc:
            try:
                tracemalloc.start()
                self._tracemalloc = True
            except Exception:
                self._tracemalloc = False
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        self._rss_start = _rss_mb()
        self._peak_rss = self._rss_start
        self._stopped = False
        return self

    def sample(self) -> None:
        rss = _rss_mb()
        if rss is None:
            return
        if self._peak_rss is None or rss > self._peak_rss:
            self._peak_rss = rss

    def stop(self) -> Dict[str, Any]:
        if self._stopped:
            return self.snapshot()
        self.sample()
        wall = time.perf_counter() - self._wall_start
        cpu = time.process_time() - self._cpu_start
        peak_python: Optional[float] = None
        if self._tracemalloc:
            try:
                _current, peak = tracemalloc.get_traced_memory()
                peak_python = peak / (1024.0 * 1024.0)
                tracemalloc.stop()
            except Exception:
                peak_python = None
            self._tracemalloc = False
        self._stopped = True
        self._wall = wall
        self._cpu = cpu
        self._peak_python = peak_python
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "wall_seconds": round(getattr(self, "_wall", 0.0), 3),
            "cpu_seconds": round(getattr(self, "_cpu", 0.0), 3),
            "peak_python_mb": (
                round(self._peak_python, 2)
                if getattr(self, "_peak_python", None) is not None
                else None
            ),
            "peak_rss_mb": (
                round(self._peak_rss, 2) if self._peak_rss is not None else None
            ),
            "ram_start_mb": (
                round(self._rss_start, 2) if self._rss_start is not None else None
            ),
        }


def document_facts(pdf_path: str, *, page_count: int) -> Dict[str, Any]:
    """Size/page facts recorded with the extraction metadata."""
    import os as _os

    size_bytes = _os.path.getsize(pdf_path)
    return {
        "source_path": str(pdf_path),
        "size_bytes": int(size_bytes),
        "size_mb": round(size_bytes / (1024.0 * 1024.0), 3),
        "page_count": int(page_count),
        "bytes_per_page": round(size_bytes / max(1, page_count), 1),
    }


def local_cost_summary(*, pages_processed: int, tables: int) -> Dict[str, Any]:
    """Cost accounting for a fully local run.

    Compute cost is reported as 0 USD because no metered API is called.
    Electricity is not metered by STE; see README for guidance.
    """
    return {
        "api_cost_usd": 0.0,
        "cost_per_1000_pages_usd": 0.0,
        "pages_processed": int(pages_processed),
        "tables": int(tables),
        "note": (
            "Local compute only for pymupdf/OCR paths. Textract AnalyzeDocument "
            "and Bedrock are metered separately when those backends are enabled."
        ),
    }


__all__ = [
    "ResourceMonitor",
    "document_facts",
    "hardware_info",
    "local_cost_summary",
]
