"""Stable table identifiers for TableRAG indexing and storage."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from app.rag.table_rag.schema import TablePayload, coerce_table_dict

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
    table: Union[Dict[str, Any], TablePayload],
    *,
    ordinal: int,
    document_id: str = "",
    page_fallback: int = 1,
) -> str:
    """Return existing ``table_id`` or assign one in-place."""
    data = coerce_table_dict(table)
    existing = str(data.get("table_id") or "").strip()
    if existing:
        if isinstance(table, TablePayload):
            table.table_id = existing
        elif isinstance(table, dict):
            table["table_id"] = existing
        return existing
    page = int(data.get("page") or page_fallback or 1)
    table_id = make_table_id(
        page=page,
        ordinal=ordinal,
        bbox=data.get("bbox"),
        headers=data.get("headers") or [],
        document_id=document_id,
    )
    if isinstance(table, TablePayload):
        table.table_id = table_id
    elif isinstance(table, dict):
        table["table_id"] = table_id
    return table_id


def parse_table_payload(raw: Any) -> Optional[TablePayload]:
    """Parse ``table_json`` from chunk/SQL metadata into ``TablePayload``."""
    if raw is None or raw == "":
        return None
    data: Any = raw
    if isinstance(raw, TablePayload):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
    if not isinstance(data, dict):
        return None
    try:
        payload = TablePayload.model_validate(data)
    except Exception:
        # Best-effort: keep unknown shapes rather than dropping the grid.
        try:
            payload = TablePayload.model_construct(**{
                k: v for k, v in data.items() if isinstance(k, str)
            })
        except Exception:
            return None
    # Normalize pages list from page / page_start / page_end when missing.
    if not payload.pages:
        pages: List[int] = []
        for key in ("page", "page_start", "page_end"):
            try:
                val = getattr(payload, key, None)
                if val is not None:
                    pages.append(int(val))
            except (TypeError, ValueError):
                pass
        payload.pages = sorted(set(pages)) if pages else []
    return payload


def table_fingerprint(table: Union[Dict[str, Any], TablePayload]) -> str:
    """Content fingerprint used to group split table parts."""
    data = coerce_table_dict(table)
    headers = [_normalize(h) for h in (data.get("headers") or [])]
    rows = data.get("rows") or []
    # Prefer full-table identity over split-part rows.
    if data.get("table_id"):
        return str(data["table_id"])
    sample_rows = [
        [_normalize(c) for c in (row or [])]
        for row in rows[:3]
    ]
    payload = {
        "page": data.get("page"),
        "pages": data.get("pages") or [],
        "headers": headers,
        "sample_rows": sample_rows,
        "bbox": data.get("bbox"),
        "n_rows": data.get("n_rows"),
        "n_cols": data.get("n_cols"),
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
