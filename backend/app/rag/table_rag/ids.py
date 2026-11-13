"""Stable table identifiers for TableRAG indexing and storage."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence


_WS_RE = re.compile(r"\s+")


def _normalize(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "")).strip()


def bbox_key(bbox: Any) -> str:
    """Compact bbox fingerprint for stable ids."""
    if bbox is None:
        return "nobbox"
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        coords = [round(float(bbox[i]), 1) for i in range(4)]
        return "x".join(str(c) for c in coords)
    if isinstance(bbox, dict):
        try:
            coords = [
                round(float(bbox.get(k, 0)), 1)
                for k in ("x0", "y0", "x1", "y1")
            ]
            return "x".join(str(c) for c in coords)
        except (TypeError, ValueError):
            pass
    return "nobbox"


def make_table_id(
    *,
    page: int,
    ordinal: int,
    bbox: Any = None,
    headers: Optional[Sequence[str]] = None,
    document_id: str = "",
) -> str:
    """Build a deterministic table id from page/order/geometry/headers.

    Format: ``t_{page}_{ordinal}_{short_hash}`` optionally prefixed with a
    short document digest so cross-document collisions stay rare.
    """
    header_blob = "|".join(_normalize(h) for h in (headers or [])[:12])
    digest = hashlib.sha1(
        f"{page}|{ordinal}|{bbox_key(bbox)}|{header_blob}".encode("utf-8")
    ).hexdigest()[:10]
    base = f"t_p{int(page)}_i{int(ordinal)}_{digest}"
    if document_id:
        doc_digest = hashlib.sha1(str(document_id).encode("utf-8")).hexdigest()[:8]
        return f"{doc_digest}_{base}"
    return base


def ensure_table_id(
    table: Dict[str, Any],
    *,
    ordinal: int,
    document_id: str = "",
    page_fallback: int = 1,
) -> str:
    """Return existing ``table_id`` or assign one in-place."""
    existing = str(table.get("table_id") or "").strip()
    if existing:
        return existing
    page = int(table.get("page") or page_fallback or 1)
    table_id = make_table_id(
        page=page,
        ordinal=ordinal,
        bbox=table.get("bbox"),
        headers=table.get("headers") or [],
        document_id=document_id,
    )
    table["table_id"] = table_id
    return table_id


def parse_table_payload(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse ``table_json`` from chunk metadata into a dict."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def table_fingerprint(table: Dict[str, Any]) -> str:
    """Content fingerprint used to dedupe split table parts."""
    headers = [_normalize(h) for h in (table.get("headers") or [])]
    rows = table.get("rows") or []
    # Prefer full-table identity over split-part rows.
    if table.get("table_id"):
        return str(table["table_id"])
    sample_rows = [
        [_normalize(c) for c in (row or [])]
        for row in rows[:3]
    ]
    payload = {
        "page": table.get("page"),
        "headers": headers,
        "sample_rows": sample_rows,
        "bbox": table.get("bbox"),
        "n_rows": table.get("n_rows"),
        "n_cols": table.get("n_cols"),
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
