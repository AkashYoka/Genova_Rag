"""Render PDF pages to PNG/JPEG bytes for OCR (never send PDF bytes)."""
from __future__ import annotations

from typing import Any, Tuple

_MAX_DETECT_BYTES = 4_500_000
_DEFAULT_DPI = 300


def render_page_image(
    page: Any,
    *,
    dpi: int = _DEFAULT_DPI,
) -> Tuple[bytes, str, int, int, int]:
    """Return (bytes, format, width, height, used_dpi)."""
    import pymupdf as fitz

    current_dpi = max(72, int(dpi))
    last_size = 0
    for _ in range(8):
        scale = current_dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img_w, img_h = int(pix.width), int(pix.height)
        png = pix.tobytes("png")
        last_size = len(png)
        if last_size <= _MAX_DETECT_BYTES:
            return png, "png", img_w, img_h, current_dpi
        try:
            jpg = pix.tobytes("jpeg", jpg_quality=85)
            if len(jpg) <= _MAX_DETECT_BYTES:
                return jpg, "jpeg", img_w, img_h, current_dpi
            last_size = len(jpg)
        except Exception:
            pass
        current_dpi = max(72, int(current_dpi * 0.75))
    raise RuntimeError(
        f"page image still too large after downscale ({last_size} bytes)"
    )


def crop_preview_png(
    page: Any,
    bbox: Tuple[float, float, float, float],
    *,
    dpi: int = 150,
    pad: float = 4.0,
) -> bytes:
    """Crop a table region preview as PNG bytes."""
    import pymupdf as fitz

    x0, y0, x1, y1 = bbox
    clip = fitz.Rect(
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(page.rect.width, x1 + pad),
        min(page.rect.height, y1 + pad),
    )
    scale = max(72, int(dpi)) / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    return pix.tobytes("png")
