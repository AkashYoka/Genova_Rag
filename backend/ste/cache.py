"""Content-addressed cache for backend extraction results.

Unchanged PDFs with unchanged backend settings are not re-converted. Only the
raw backend output (grids + cell geometry) is cached; reconstruction,
validation and export always run again so fixes in post-processing take effect
immediately.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Bump when backend post-processing changes so stale extraction is not reused.
CACHE_SCHEMA = 6


def default_cache_dir() -> Path:
    env = os.environ.get("STE_CACHE_DIR")
    if env:
        return Path(env)
    return _PROJECT_ROOT / ".ste_cache"


def file_digest(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of file contents (streamed)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class ExtractionCache:
    """JSON cache keyed by PDF content + backend + options + pages."""

    def __init__(self, root: Optional[str | Path] = None, *, enabled: bool = True) -> None:
        self.root = Path(root) if root else default_cache_dir()
        self.enabled = bool(enabled)
        self._digests: Dict[tuple[str, int, int], str] = {}
        self.hits = 0
        self.misses = 0

    def document_digest(self, pdf_path: str | Path) -> str:
        """Digest of the PDF, memoized on path/size/mtime within a process."""
        path = Path(pdf_path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, int(stat.st_mtime))
        if key not in self._digests:
            self._digests[key] = file_digest(path)
        return self._digests[key]

    def make_key(
        self,
        pdf_path: str | Path,
        *,
        backend: str,
        options: Dict[str, Any],
        pages: Sequence[int],
    ) -> str:
        """Deterministic cache key for one extraction request."""
        payload = {
            "schema": CACHE_SCHEMA,
            "pdf_sha256": self.document_digest(pdf_path),
            "backend": backend,
            "options": options,
            "pages": sorted(int(p) for p in pages),
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self.root / f"{key}.json"
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.hits += 1
            return data.get("extraction") or None
        except Exception as exc:
            logger.warning("Ignoring unreadable extraction cache %s: %s", path, exc)
            self.misses += 1
            return None

    def save(
        self,
        key: str,
        extraction: Dict[str, Any],
        *,
        pdf_path: str | Path,
        backend: str,
        options: Dict[str, Any],
        pages: Sequence[int],
        elapsed_seconds: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": 1,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "pdf_path": str(Path(pdf_path).resolve()),
                "backend": backend,
                "options": options,
                "pages": sorted(int(p) for p in pages),
                "elapsed_seconds": float(elapsed_seconds),
                "extraction": extraction,
            }
            target = self.root / f"{key}.json"
            tmp = target.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(target)
        except Exception as exc:
            logger.warning("Could not write extraction cache: %s", exc)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "root": str(self.root),
            "hits": self.hits,
            "misses": self.misses,
        }

    def clear(self) -> int:
        """Delete cached extraction files; returns the count removed."""
        if not self.root.exists():
            return 0
        removed = 0
        for path in self.root.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed


__all__ = [
    "ExtractionCache",
    "default_cache_dir",
    "file_digest",
]
