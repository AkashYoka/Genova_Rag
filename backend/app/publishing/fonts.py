"""Register Unicode-capable fonts for Genova PDF publishing.

Falls back to Helvetica/Courier when system TTFs are unavailable (e.g. Linux CI).
Prefers fonts with broad Unicode coverage (CJK + Latin) when present.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Logical font names used throughout the publishing package
SANS = "GenovaSans"
SANS_BOLD = "GenovaSans-Bold"
SANS_ITALIC = "GenovaSans-Italic"
SANS_BOLD_ITALIC = "GenovaSans-BoldItalic"
MONO = "GenovaMono"

_WIN = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

# (path, subfontIndex) — Yu Gothic / MS YaHei cover Latin + CJK
_CANDIDATES: Dict[str, List[Tuple[str, int]]] = {
    SANS: [
        (os.path.join(_WIN, "YuGothR.ttc"), 0),
        (os.path.join(_WIN, "msyh.ttc"), 0),
        (os.path.join(_WIN, "arial.ttf"), 0),
        (os.path.join(_WIN, "segoeui.ttf"), 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 0),
    ],
    SANS_BOLD: [
        (os.path.join(_WIN, "YuGothB.ttc"), 0),
        (os.path.join(_WIN, "msyhbd.ttc"), 0),
        (os.path.join(_WIN, "arialbd.ttf"), 0),
        (os.path.join(_WIN, "segoeuib.ttf"), 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0),
    ],
    SANS_ITALIC: [
        (os.path.join(_WIN, "ariali.ttf"), 0),
        (os.path.join(_WIN, "segoeuii.ttf"), 0),
        (os.path.join(_WIN, "YuGothR.ttc"), 0),  # no italic — reuse regular
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 0),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf", 0),
    ],
    SANS_BOLD_ITALIC: [
        (os.path.join(_WIN, "arialbi.ttf"), 0),
        (os.path.join(_WIN, "segoeuiz.ttf"), 0),
        (os.path.join(_WIN, "YuGothB.ttc"), 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf", 0),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 0),
    ],
    MONO: [
        (os.path.join(_WIN, "consola.ttf"), 0),
        (os.path.join(_WIN, "cour.ttf"), 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
        ("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 0),
    ],
}

_FALLBACK = {
    SANS: "Helvetica",
    SANS_BOLD: "Helvetica-Bold",
    SANS_ITALIC: "Helvetica-Oblique",
    SANS_BOLD_ITALIC: "Helvetica-BoldOblique",
    MONO: "Courier",
}


def _first_existing(candidates: List[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
    for path, index in candidates:
        if path and os.path.isfile(path):
            return path, index
    return None


@lru_cache(maxsize=1)
def ensure_fonts() -> Dict[str, str]:
    """Register TTFs once; return mapping logical name → registered name."""
    resolved: Dict[str, str] = {}
    for logical, candidates in _CANDIDATES.items():
        found = _first_existing(candidates)
        if found:
            path, index = found
            try:
                if path.lower().endswith(".ttc"):
                    pdfmetrics.registerFont(TTFont(logical, path, subfontIndex=index))
                else:
                    pdfmetrics.registerFont(TTFont(logical, path))
                resolved[logical] = logical
                continue
            except Exception:
                pass
        resolved[logical] = _FALLBACK[logical]
    return resolved


def font(logical: str) -> str:
    """Resolve a logical publishing font to a ReportLab font name."""
    return ensure_fonts().get(logical, _FALLBACK.get(logical, "Helvetica"))
