"""AWS Textract DetectDocumentText on page images only (never PDF bytes)."""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

from ste.models import BBox, Word, normalize_ws

logger = logging.getLogger(__name__)

_client = None


def get_client(region: Optional[str] = None):
    global _client
    if _client is not None:
        return _client
    import boto3
    from botocore.config import Config

    region = (
        region
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    cfg = Config(
        read_timeout=60,
        connect_timeout=10,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    _client = boto3.client("textract", region_name=region, config=cfg)
    return _client


def reset_client() -> None:
    global _client
    _client = None


def detect_document_text(
    image_bytes: bytes,
    *,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Call DetectDocumentText with retry/throttle."""
    client = get_client(region=region)
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            return client.detect_document_text(Document={"Bytes": image_bytes})
        except Exception as exc:
            last_exc = exc
            wait = (2**attempt) + random.random()
            logger.warning(
                "Textract attempt %s failed: %s; sleep %.1fs",
                attempt + 1,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"DetectDocumentText failed: {last_exc}") from last_exc


def words_from_blocks(
    blocks: List[Dict[str, Any]],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    image_width: int = 0,
    image_height: int = 0,
    min_confidence: float = 0.0,
) -> List[Word]:
    words: List[Word] = []
    for block in blocks:
        if block.get("BlockType") != "WORD":
            continue
        text = normalize_ws(str(block.get("Text") or ""))
        if not text:
            continue
        conf = float(block.get("Confidence") or 0.0)
        if conf < min_confidence:
            continue
        geometry = block.get("Geometry") or {}
        box = geometry.get("BoundingBox") or {}
        left = float(box.get("Left") or 0.0)
        top = float(box.get("Top") or 0.0)
        width = float(box.get("Width") or 0.0)
        height = float(box.get("Height") or 0.0)
        bbox = BBox(
            left * page_width,
            top * page_height,
            (left + width) * page_width,
            (top + height) * page_height,
        )
        if bbox.area <= 0:
            continue
        polygon_px: List[int] = []
        if image_width > 0 and image_height > 0:
            for point in (geometry.get("Polygon") or [])[:4]:
                try:
                    polygon_px.extend(
                        [
                            int(float(point.get("X") or 0.0) * image_width),
                            int(float(point.get("Y") or 0.0) * image_height),
                        ]
                    )
                except Exception:
                    continue
            while len(polygon_px) < 8 and polygon_px:
                polygon_px.extend(polygon_px[-2:])
            polygon_px = polygon_px[:8]
        words.append(
            Word(
                text=text,
                bbox=bbox,
                page=page_number,
                confidence=conf,
                polygon_px=polygon_px,
            )
        )
    return words
