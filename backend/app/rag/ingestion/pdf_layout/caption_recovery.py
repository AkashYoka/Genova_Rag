"""Domain-agnostic table caption recovery from page text near a table bbox.

Docling TableFormer often returns empty ``caption_text`` for matrices even when
the PDF has an explicit title above the grid (``Table 1. …``, ``Figure 2: …``).
This module recovers those titles from page geometry so TableRAG can scope-rank
sibling matrices without domain hardcoding.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Sequence, Tuple

from app.rag.ingestion.pdf_layout.types import BBox, normalize_ws

logger = logging.getLogger(__name__)

# Leading label for titled floats / matrices across domains.
_CAPTION_LEAD_RE = re.compile(
    r"(?is)^\s*((?:Table|Tbl\.?|Figure|Fig\.?|Exhibit|Annex|Appendix|"
    r"Schedule|Chart|Plate|Map)\s+[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\s*[.:)\-–—]\s*|\s+)(.*)$"
)
_CONTINUATION_RE = re.compile(
    r"(?is)^\s*[A-Za-z0-9(][\w\s,;:/\-–—()'\"%]{2,160}$"
)
# Reject obvious body/header grid lines mistaken for titles.
_NOT_CAPTION_RE = re.compile(
    r"(?i)^\s*(?:pretreatment|body\s+weight|dosing\s+freq|dose\s*\(mg\)|"
    r"column\s+\d+|page\s+\d+$|\d+\s*$)"
)


def _block_text(raw: Any) -> str:
    if isinstance(raw, (list, tuple)) and len(raw) >= 5:
        return normalize_ws(str(raw[4] or ""))
    return normalize_ws(str(raw or ""))


def _is_plausible_caption_start(text: str) -> bool:
    if not text or len(text) < 6:
        return False
    if _NOT_CAPTION_RE.search(text):
        return False
    return bool(_CAPTION_LEAD_RE.match(text))


def _merge_caption_lines(lines: Sequence[str], *, max_chars: int = 320) -> str:
    parts: List[str] = []
    for line in lines:
        text = normalize_ws(line)
        if not text:
            continue
        if parts and _NOT_CAPTION_RE.search(text):
            break
        if parts and not _CONTINUATION_RE.match(text):
            # Stop when the next line looks like a new section / grid header.
            if re.match(r"(?i)^\s*\d+(\.\d+)*\s+[A-Z]", text):
                break
            if len(text) < 4:
                break
        parts.append(text)
        joined = normalize_ws(" ".join(parts))
        if len(joined) >= max_chars:
            return joined[:max_chars].rstrip()
    return normalize_ws(" ".join(parts))[:max_chars]


def recover_caption_from_page_blocks(
    blocks: Sequence[Any],
    table_bbox: BBox,
    *,
    max_gap_pt: float = 96.0,
    max_lines: int = 4,
) -> str:
    """Pick a titled caption from text blocks immediately above ``table_bbox``."""
    if not blocks or table_bbox is None:
        return ""
    try:
        top = float(table_bbox.y0)
        left = float(table_bbox.x0)
        right = float(table_bbox.x1)
    except Exception:
        return ""

    candidates: List[Tuple[float, float, str]] = []
    for raw in blocks:
        try:
            if isinstance(raw, (list, tuple)) and len(raw) >= 5:
                x0, y0, x1, y1 = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
                text = _block_text(raw)
            else:
                continue
        except Exception:
            continue
        if not text:
            continue
        # Only consider text above the table, within a vertical gap.
        if y1 > top + 2.0:
            continue
        if top - y1 > max_gap_pt:
            continue
        # Horizontal overlap with the table (loose).
        overlap = min(right, x1) - max(left, x0)
        if overlap < -40:
            continue
        candidates.append((y0, x0, text))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], item[1]))

    # Prefer the nearest caption-start above the table; allow 1–N continuation lines.
    for idx in range(len(candidates) - 1, -1, -1):
        _y0, _x0, text = candidates[idx]
        if not _is_plausible_caption_start(text):
            continue
        lines = [text]
        for j in range(idx + 1, min(len(candidates), idx + max_lines)):
            nxt = candidates[j][2]
            if _is_plausible_caption_start(nxt):
                break
            lines.append(nxt)
        caption = _merge_caption_lines(lines)
        if caption:
            return caption
    return ""


def _bbox_is_full_page(bbox: BBox, page_width: float, page_height: float) -> bool:
    try:
        area = max(0.0, float(bbox.x1) - float(bbox.x0)) * max(
            0.0, float(bbox.y1) - float(bbox.y0)
        )
        page_area = max(1.0, float(page_width) * float(page_height))
        return (area / page_area) >= 0.85 or (
            float(bbox.x0) <= 1.0
            and float(bbox.y0) <= 1.0
            and float(bbox.x1) >= page_width - 1.0
            and float(bbox.y1) >= page_height - 1.0
        )
    except Exception:
        return False


def recover_caption_from_page_top(
    blocks: Sequence[Any],
    *,
    max_lines: int = 4,
) -> str:
    """First titled float on the page (used when table bbox is full-page/unknown)."""
    ordered: List[Tuple[float, float, str]] = []
    for raw in blocks or []:
        try:
            if not (isinstance(raw, (list, tuple)) and len(raw) >= 5):
                continue
            text = _block_text(raw)
            if not text:
                continue
            ordered.append((float(raw[1]), float(raw[0]), text))
        except Exception:
            continue
    ordered.sort(key=lambda item: (item[0], item[1]))
    for idx, (_y, _x, text) in enumerate(ordered):
        if not _is_plausible_caption_start(text):
            continue
        lines = [text]
        for j in range(idx + 1, min(len(ordered), idx + max_lines)):
            nxt = ordered[j][2]
            if _is_plausible_caption_start(nxt):
                break
            if _NOT_CAPTION_RE.search(nxt):
                break
            lines.append(nxt)
        caption = _merge_caption_lines(lines)
        if caption:
            return caption
    return ""


def recover_caption_from_pdf_page(
    filepath: str,
    page_number: int,
    table_bbox: Optional[BBox] = None,
    *,
    max_gap_pt: float = 96.0,
) -> str:
    """Open ``filepath`` page and recover a caption above the table bbox.

    When the stored bbox is missing or full-page (common Docling provenance
    failure), fall back to the first titled line on the page.
    """
    if not filepath or page_number < 1:
        return ""
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        logger.debug("caption recovery skipped (no fitz): %s", exc)
        return ""
    try:
        doc = fitz.open(filepath)
    except Exception as exc:
        logger.debug("caption recovery open failed: %s", exc)
        return ""
    try:
        if page_number > doc.page_count:
            return ""
        page = doc[page_number - 1]
        blocks = page.get_text("blocks") or []
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)
        caption = ""
        if table_bbox is not None and not _bbox_is_full_page(table_bbox, page_w, page_h):
            caption = recover_caption_from_page_blocks(
                blocks, table_bbox, max_gap_pt=max_gap_pt
            )
        if not caption:
            caption = recover_caption_from_page_top(blocks)
        return caption
    except Exception as exc:
        logger.debug("caption recovery page %s failed: %s", page_number, exc)
        return ""
    finally:
        try:
            doc.close()
        except Exception:
            pass


def recover_caption_from_docling_neighbors(
    document: Any,
    table: Any,
    *,
    max_lookback: int = 6,
) -> str:
    """Fallback: walk Docling items preceding the table for a titled line."""
    if document is None or table is None:
        return ""
    items: List[Any] = []
    try:
        if hasattr(document, "iterate_items"):
            items = [item for item, _path in document.iterate_items()]
        elif hasattr(document, "texts") and document.texts:
            # No stable order vs tables — skip.
            return ""
    except Exception:
        return ""
    if not items:
        return ""
    try:
        idx = next(i for i, it in enumerate(items) if it is table)
    except StopIteration:
        # Identity may differ after export; match by page + rough position.
        return ""
    collected: List[str] = []
    for j in range(idx - 1, max(-1, idx - 1 - max_lookback), -1):
        item = items[j]
        label = str(getattr(item, "label", "") or "").lower()
        if "table" in label:
            break
        text = ""
        for attr in ("text", "orig"):
            val = getattr(item, attr, None)
            if callable(val):
                continue
            if val:
                text = normalize_ws(str(val))
                break
        if not text:
            continue
        collected.insert(0, text)
        if _is_plausible_caption_start(text):
            return _merge_caption_lines(collected)
    # If we collected lines ending with a caption start earlier, re-scan.
    for i, text in enumerate(collected):
        if _is_plausible_caption_start(text):
            return _merge_caption_lines(collected[i:])
    return ""


def enrich_table_caption(
    table_data: Any,
    *,
    filepath: str = "",
    document: Any = None,
    docling_table: Any = None,
) -> str:
    """Fill ``table_data.caption`` when empty; return the caption used."""
    existing = normalize_ws(str(getattr(table_data, "caption", "") or ""))
    if existing:
        return existing

    caption = ""
    if document is not None and docling_table is not None:
        caption = recover_caption_from_docling_neighbors(document, docling_table)
    if not caption and filepath:
        page = int(getattr(table_data, "page", 0) or 0)
        bbox = getattr(table_data, "bbox", None)
        if page and bbox is not None:
            caption = recover_caption_from_pdf_page(filepath, page, bbox)
    caption = normalize_ws(caption)
    if caption:
        try:
            table_data.caption = caption
            prov = getattr(table_data, "provenance", None)
            if isinstance(prov, dict):
                prov = dict(prov)
                prov["caption_recovered"] = True
                prov["caption_source"] = "page_text_or_docling_neighbor"
                table_data.provenance = prov
        except Exception:
            pass
    return caption
