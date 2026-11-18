"""Local Tesseract OCR fallback (page images → positioned words)."""
from __future__ import annotations

import io
import logging
from typing import List, Tuple

from ste.models import BBox, Word, normalize_ws

logger = logging.getLogger(__name__)


def words_from_image(
    image_bytes: bytes,
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> List[Word]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract and Pillow required for tesseract OCR backend"
        ) from exc

    img = Image.open(io.BytesIO(image_bytes))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n = len(data.get("text") or [])
    sx = page_width / max(1, image_width)
    sy = page_height / max(1, image_height)
    words: List[Word] = []
    for i in range(n):
        text = normalize_ws(str(data["text"][i] or ""))
        if not text:
            continue
        conf_raw = data.get("conf", ["-1"])[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.0
        if conf < 0:
            continue
        x, y, w, h = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        bbox = BBox(x * sx, y * sy, (x + w) * sx, (y + h) * sy)
        if bbox.area <= 0:
            continue
        words.append(
            Word(text=text, bbox=bbox, page=page_number, confidence=conf)
        )
    return words
