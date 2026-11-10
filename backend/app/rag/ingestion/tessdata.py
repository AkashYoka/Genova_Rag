"""Resolve Tesseract language-data directory for Windows/Scoop and Linux installs."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _has_eng(path: Path) -> bool:
    return (path / "eng.traineddata").is_file()


def resolve_tessdata_dir(explicit: Optional[str] = None) -> str:
    """Return a directory that contains ``eng.traineddata``, or empty string.

    Preference order:
    1. Explicit argument / ``PDF_OCR_TESSDATA_DIR``
    2. ``TESSDATA_PREFIX`` env
    3. Common Scoop / local install locations
    4. ``tesseract --list-langs`` sibling ``tessdata`` folder
    """
    candidates: list[Path] = []

    for raw in (
        explicit,
        os.environ.get("PDF_OCR_TESSDATA_DIR"),
        os.environ.get("TESSDATA_PREFIX"),
    ):
        if raw and str(raw).strip():
            candidates.append(Path(str(raw).strip()))

    home = Path.home()
    scoop = home / "scoop" / "apps"
    candidates.extend(
        [
            scoop / "tesseract-languages" / "current",
            scoop / "tesseract-languages" / "current" / "tessdata",
            scoop / "tesseract" / "current" / "tessdata",
            Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
            Path("/usr/share/tesseract-ocr/5/tessdata"),
            Path("/usr/share/tesseract-ocr/4.00/tessdata"),
            Path("/usr/share/tessdata"),
        ]
    )

    try:
        import shutil
        import subprocess

        binary = shutil.which("tesseract")
        if binary:
            binary_path = Path(binary).resolve()
            candidates.append(binary_path.parent / "tessdata")
            # Scoop shim → real install
            if "shims" in binary_path.parts:
                candidates.append(scoop / "tesseract" / "current" / "tessdata")
    except Exception:
        pass

    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if _has_eng(resolved):
            return str(resolved)
        nested = resolved / "tessdata"
        if _has_eng(nested):
            return str(nested)

    return ""


def apply_tessdata_env(explicit: Optional[str] = None) -> str:
    """Set ``TESSDATA_PREFIX`` when a usable language pack is found."""
    tessdata = resolve_tessdata_dir(explicit)
    if tessdata:
        current = (os.environ.get("TESSDATA_PREFIX") or "").strip()
        if current != tessdata:
            os.environ["TESSDATA_PREFIX"] = tessdata
            logger.info("Using Tesseract language data at %s", tessdata)
    return tessdata
