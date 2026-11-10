"""Page-level PDF classification based on extraction quality, not char count alone.

The classifier separates four production cases:

* ``native``  — usable selectable text layer.
* ``scanned`` — image-only (or near-image-only) page; OCR required.
* ``corrupt`` — a text layer exists but is garbled, sparse, or unreliable.
* ``blank``   — no text and no meaningful image coverage.

A page is never considered native merely because a handful of characters were
extracted. Character volume, alphanumeric density, meaningful-word ratio,
garbage/replacement character ratio, and image coverage all contribute.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from app.config import get_settings

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_COMMON_WORDS = {
    "a", "about", "above", "after", "all", "also", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "between", "both", "but", "by", "can",
    "company", "could", "do", "does", "each", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "if", "in", "into", "is", "it", "its", "may", "more", "most",
    "must", "no", "not", "of", "on", "one", "only", "or", "other", "our", "out", "over",
    "page", "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "two", "under", "up", "was", "we", "were", "what",
    "when", "which", "while", "who", "will", "with", "would", "year", "you",
}
_GARBAGE_CHARS = set("\ufffd\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\u000b\u000c")


def _is_meaningful_word(word: str) -> bool:
    lowered = word.lower()
    if lowered in _COMMON_WORDS:
        return True
    if len(lowered) < 3:
        return False
    vowels = sum(1 for ch in lowered if ch in "aeiouy")
    if vowels == 0:
        return False
    # Long consonant runs are a strong signal of OCR/text-layer gibberish.
    return not re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", lowered)


def compute_text_metrics(text: str, *, page_area: float = 0.0, image_coverage: float = 0.0) -> Dict[str, Any]:
    raw = text or ""
    stripped = raw.strip()
    compact = re.sub(r"\s+", "", stripped)
    chars = len(compact)
    words = _WORD_RE.findall(stripped)
    alnum = sum(1 for ch in compact if ch.isalnum())
    garbage = sum(1 for ch in compact if ch in _GARBAGE_CHARS or (ord(ch) < 32 and ch not in "\n\t\r"))
    meaningful = sum(1 for w in words if _is_meaningful_word(w))
    digits = sum(1 for ch in compact if ch.isdigit())
    return {
        "chars": chars,
        "words": len(words),
        "lines": len([line for line in raw.splitlines() if line.strip()]),
        "alnum_ratio": round(alnum / chars, 4) if chars else 0.0,
        "garbage_ratio": round(garbage / chars, 4) if chars else 0.0,
        "meaningful_word_ratio": round(meaningful / len(words), 4) if words else 0.0,
        "digit_ratio": round(digits / chars, 4) if chars else 0.0,
        "chars_per_1000pt2": round(chars / (page_area / 1000.0), 2) if page_area else 0.0,
        "image_coverage": round(max(0.0, min(1.0, image_coverage)), 4),
    }


def classify_page(
    text: str,
    *,
    page_area: float = 0.0,
    image_coverage: float = 0.0,
    min_chars: int | None = None,
) -> Dict[str, Any]:
    """Return ``{label, needs_ocr, confidence, reasons, metrics}`` for one page."""
    settings = get_settings()
    min_chars = settings.PDF_OCR_MIN_CHARS if min_chars is None else min_chars
    metrics = compute_text_metrics(text, page_area=page_area, image_coverage=image_coverage)
    chars = metrics["chars"]
    reasons: List[str] = []

    if chars == 0:
        if image_coverage >= 0.15:
            reasons.append("no selectable text and page is image-backed")
            return _result("scanned", True, 0.99, reasons, metrics)
        reasons.append("no selectable text and no meaningful image")
        return _result("blank", False, 0.99, reasons, metrics)

    if chars < 20 and image_coverage >= 0.15:
        reasons.append(f"only {chars} characters with {image_coverage:.0%} image coverage")
        return _result("scanned", True, 0.97, reasons, metrics)

    if chars < min_chars and image_coverage >= settings.PDF_OCR_IMAGE_COVERAGE_THRESHOLD:
        reasons.append(f"sparse text ({chars} chars) under image-heavy page")
        return _result("scanned", True, 0.9, reasons, metrics)

    corrupt_reasons: List[str] = []
    if metrics["alnum_ratio"] < settings.PDF_OCR_MIN_ALNUM_RATIO:
        corrupt_reasons.append(f"low alphanumeric ratio {metrics['alnum_ratio']}")
    if metrics["garbage_ratio"] > settings.PDF_OCR_MAX_GARBAGE_RATIO:
        corrupt_reasons.append(f"high garbage ratio {metrics['garbage_ratio']}")
    if metrics["words"] >= 8 and metrics["meaningful_word_ratio"] < 0.45:
        corrupt_reasons.append(
            f"low meaningful-word ratio {metrics['meaningful_word_ratio']}"
        )

    if corrupt_reasons:
        reasons.extend(corrupt_reasons)
        needs_ocr = image_coverage >= 0.15 and settings.PDF_OCR_ENABLED
        if needs_ocr:
            reasons.append("image-backed page can be recovered with OCR")
        return _result("corrupt", needs_ocr, 0.8, reasons, metrics)

    if chars < min_chars and image_coverage < 0.15:
        reasons.append(f"sparse selectable text ({chars} chars) and no image fallback")
        return _result("corrupt", False, 0.6, reasons, metrics)

    reasons.append("usable selectable text layer")
    return _result("native", False, 0.95, reasons, metrics)


def _result(label: str, needs_ocr: bool, confidence: float, reasons: List[str], metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": label,
        "needs_ocr": needs_ocr,
        "confidence": confidence,
        "reasons": reasons,
        "metrics": metrics,
    }


def classify_document(pages: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate page classifications into text_based / scanned / hybrid / empty."""
    labels = [p.get("classification", "unknown") for p in pages]
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ocr_pages = sum(1 for p in pages if p.get("strategy") == "ocr")
    native_pages = sum(1 for p in pages if p.get("strategy") == "native")
    if not labels:
        kind = "empty"
    elif ocr_pages == 0:
        kind = "text_based"
    elif native_pages == 0:
        kind = "scanned"
    else:
        kind = "hybrid"
    return {
        "document_type": kind,
        "page_counts": counts,
        "native_pages": native_pages,
        "ocr_pages": ocr_pages,
        "failed_pages": sum(1 for p in pages if p.get("status") == "failed"),
    }
