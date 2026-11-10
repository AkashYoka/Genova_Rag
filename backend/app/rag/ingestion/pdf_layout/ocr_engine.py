"""OCR backends that return positioned words, not flat text.

Tesseract is the primary engine because it is already a project dependency,
runs on CPU, provides word boxes plus confidence, and needs no model server.
EasyOCR is retained as an optional alternative (explicit configuration).
RapidOCR is used automatically only when installed and Tesseract is absent;
this keeps deployments portable without adding a mandatory heavyweight engine.
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional, Tuple

import pymupdf as fitz

from app.config import get_settings
from app.rag.ingestion.pdf_layout.types import BBox, TextWord

logger = logging.getLogger(__name__)


class OcrError(RuntimeError):
    """Recoverable OCR failure for a single page."""


class OcrUnavailableError(OcrError):
    """No configured OCR backend is available."""


def tesseract_available() -> bool:
    try:
        from app.rag.ingestion.tessdata import apply_tessdata_env
        from app.config import get_settings

        apply_tessdata_env(get_settings().PDF_OCR_TESSDATA_DIR)
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def rapidocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


def available_backends() -> List[str]:
    backends: List[str] = []
    if tesseract_available():
        backends.append("tesseract")
    if rapidocr_available():
        backends.append("rapidocr")
    if easyocr_available():
        backends.append("easyocr")
    return backends


def resolve_backend(preferred: Optional[str] = None) -> str:
    settings = get_settings()
    requested = (preferred or settings.PDF_OCR_BACKEND or "auto").strip().lower()
    if requested in {"", "auto"}:
        for backend in ("tesseract", "rapidocr", "easyocr"):
            if backend == "tesseract" and tesseract_available():
                return backend
            if backend == "rapidocr" and rapidocr_available():
                return backend
            if backend == "easyocr" and easyocr_available():
                return backend
        raise OcrUnavailableError(
            "No OCR backend is available. Install Tesseract (tesseract-ocr + eng "
            "language data) or set PDF_OCR_BACKEND=easyocr with easyocr installed."
        )
    if requested == "tesseract" and not tesseract_available():
        raise OcrUnavailableError("Tesseract binary is not installed or not on PATH")
    if requested == "rapidocr" and not rapidocr_available():
        raise OcrUnavailableError("rapidocr-onnxruntime is not installed")
    if requested == "easyocr" and not easyocr_available():
        raise OcrUnavailableError("easyocr is not installed")
    return requested


def _render_page(page: fitz.Page, dpi: int) -> Tuple[Any, int, int]:
    """Render a page to a Pillow image; return image plus pixel dimensions."""
    from PIL import Image

    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image, pix.width, pix.height


def _to_pdf_bbox(left: float, top: float, width: float, height: float, page: fitz.Page, px_w: int, px_h: int) -> BBox:
    sx = float(page.rect.width) / max(1, px_w)
    sy = float(page.rect.height) / max(1, px_h)
    return BBox(left * sx, top * sy, (left + width) * sx, (top + height) * sy)


def _tesseract_words(
    image: Any,
    page: fitz.Page,
    px_w: int,
    px_h: int,
    *,
    lang: str,
    psm: int,
    timeout: float,
    min_confidence: float,
) -> Tuple[List[TextWord], float]:
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise OcrUnavailableError("pytesseract is not installed") from exc

    settings = get_settings()
    config = f"--psm {int(psm)}"
    # Tesseract reads language data from TESSDATA_PREFIX. Passing a quoted
    # --tessdata-dir through pytesseract's config string breaks on Windows
    # because the quotes become part of the path. Auto-discover Scoop /
    # system language packs when PDF_OCR_TESSDATA_DIR is unset.
    from app.rag.ingestion.tessdata import apply_tessdata_env

    tessdata = apply_tessdata_env(settings.PDF_OCR_TESSDATA_DIR)
    if not tessdata:
        raise OcrUnavailableError(
            "Tesseract language data (eng.traineddata) not found. "
            "Set PDF_OCR_TESSDATA_DIR or install tesseract-languages."
        )

    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config,
            timeout=timeout,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError as exc:
        raise OcrError(f"Tesseract failed: {exc}") from exc
    except RuntimeError as exc:
        message = str(exc)
        if "timeout" in message.lower() or "timed out" in message.lower():
            raise OcrError(f"Tesseract timed out after {timeout:.0f}s") from exc
        raise OcrError(f"Tesseract failed: {message}") from exc
    except Exception as exc:
        raise OcrError(f"Tesseract failed: {exc}") from exc

    words: List[TextWord] = []
    confidences: List[float] = []
    for idx, raw_text in enumerate(data.get("text", [])):
        text = (raw_text or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][idx])
        except Exception:
            conf = 0.0
        if conf < min_confidence:
            continue
        left = float(data["left"][idx])
        top = float(data["top"][idx])
        width = float(data["width"][idx])
        height = float(data["height"][idx])
        if width <= 0 or height <= 0:
            continue
        words.append(
            TextWord(
                text=text,
                bbox=_to_pdf_bbox(left, top, width, height, page, px_w, px_h),
                page=page.number + 1,
                confidence=conf,
            )
        )
        confidences.append(conf)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return words, mean_conf


def _rapidocr_words(image: Any, page: fitz.Page, px_w: int, px_h: int) -> Tuple[List[TextWord], float]:
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise OcrUnavailableError("rapidocr-onnxruntime is not installed") from exc

    global _rapidocr_engine  # noqa: PLW0603
    if "_rapidocr_engine" not in globals():
        _rapidocr_engine = RapidOCR()

    result, _elapsed = _rapidocr_engine(np.array(image.convert("RGB")))
    words: List[TextWord] = []
    confidences: List[float] = []
    for item in result or []:
        try:
            box, text, score = item
            text = str(text or "").strip()
            if not text:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            sx = float(page.rect.width) / max(1, px_w)
            sy = float(page.rect.height) / max(1, px_h)
            words.append(
                TextWord(
                    text=text,
                    bbox=BBox(min(xs) * sx, min(ys) * sy, max(xs) * sx, max(ys) * sy),
                    page=page.number + 1,
                    confidence=float(score) * 100.0,
                )
            )
            confidences.append(float(score) * 100.0)
        except Exception:
            continue
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return words, mean_conf


def _easyocr_words(image: Any, page: fitz.Page, px_w: int, px_h: int) -> Tuple[List[TextWord], float]:
    try:
        import numpy as np
        import easyocr
    except ImportError as exc:
        raise OcrUnavailableError("easyocr is not installed") from exc

    global _easyocr_engine  # noqa: PLW0603
    if "_easyocr_engine" not in globals():
        _easyocr_engine = easyocr.Reader(["en"], gpu=False, verbose=False)

    results = _easyocr_engine.readtext(np.array(image.convert("RGB")), detail=1, paragraph=False)
    words: List[TextWord] = []
    confidences: List[float] = []
    for box, text, conf in results or []:
        text = str(text or "").strip()
        if not text:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        sx = float(page.rect.width) / max(1, px_w)
        sy = float(page.rect.height) / max(1, px_h)
        words.append(
            TextWord(
                text=text,
                bbox=BBox(min(xs) * sx, min(ys) * sy, max(xs) * sx, max(ys) * sy),
                page=page.number + 1,
                confidence=float(conf) * 100.0,
            )
        )
        confidences.append(float(conf) * 100.0)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return words, mean_conf


def _run_with_timeout(fn, timeout: float):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            raise OcrError(f"OCR timed out after {timeout:.0f}s") from exc


def ocr_page(
    page: fitz.Page,
    *,
    dpi: Optional[int] = None,
    backend: Optional[str] = None,
    timeout: Optional[float] = None,
    lang: Optional[str] = None,
    psm: Optional[int] = None,
) -> Dict[str, Any]:
    """OCR one page and return positioned words with confidence and timing."""
    settings = get_settings()
    dpi = int(dpi or settings.PDF_OCR_DPI)
    timeout = float(timeout or settings.PDF_OCR_TIMEOUT_SECONDS)
    lang = lang or settings.PDF_OCR_LANG
    psm = int(psm or settings.PDF_OCR_PSM)
    selected = resolve_backend(backend)

    started = time.perf_counter()
    image, px_w, px_h = _render_page(page, dpi)

    if selected == "tesseract":
        words, mean_conf = _tesseract_words(
            image,
            page,
            px_w,
            px_h,
            lang=lang,
            psm=psm,
            timeout=timeout,
            min_confidence=settings.PDF_OCR_MIN_CONFIDENCE,
        )
    elif selected == "rapidocr":
        words, mean_conf = _run_with_timeout(
            lambda: _rapidocr_words(image, page, px_w, px_h), timeout
        )
    elif selected == "easyocr":
        words, mean_conf = _run_with_timeout(
            lambda: _easyocr_words(image, page, px_w, px_h), timeout
        )
    else:  # pragma: no cover - resolve_backend guards this
        raise OcrUnavailableError(f"Unsupported OCR backend: {selected}")

    return {
        "words": words,
        "mean_confidence": round(mean_conf, 2),
        "backend": selected,
        "dpi": dpi,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "pixel_size": [px_w, px_h],
    }


def ocr_status() -> Dict[str, Any]:
    from app.rag.ingestion.tessdata import apply_tessdata_env

    settings = get_settings()
    tessdata = apply_tessdata_env(settings.PDF_OCR_TESSDATA_DIR)
    return {
        "configured_backend": settings.PDF_OCR_BACKEND,
        "available_backends": available_backends(),
        "dpi": settings.PDF_OCR_DPI,
        "timeout_s": settings.PDF_OCR_TIMEOUT_SECONDS,
        "tesseract_binary": shutil.which("tesseract") or "",
        "tessdata_prefix": tessdata or settings.PDF_OCR_TESSDATA_DIR or os.environ.get("TESSDATA_PREFIX", ""),
    }
