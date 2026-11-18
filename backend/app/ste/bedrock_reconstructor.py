"""AWS Bedrock semantic reconstruction (universal, domain-agnostic)."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ste.models import ExtractedTable, normalize_ws, pad_row
from ste.prompts.universal_reconstruction import (
    UNIVERSAL_SYSTEM_PROMPT,
    build_user_payload,
)

logger = logging.getLogger(__name__)

_bedrock_client = None

DEFAULT_MODEL_ID = (
    os.environ.get("STE_BEDROCK_MODEL_ID")
    or os.environ.get("BEDROCK_MODEL_ID")
    or "us.amazon.nova-2-lite-v1:0"
)


def get_bedrock_client(region: Optional[str] = None):
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client
    import boto3
    from botocore.config import Config

    region = (
        region
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    cfg = Config(
        read_timeout=120,
        connect_timeout=15,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    _bedrock_client = boto3.client("bedrock-runtime", region_name=region, config=cfg)
    return _bedrock_client


def reset_bedrock_client() -> None:
    global _bedrock_client
    _bedrock_client = None


def _extract_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty Bedrock response")
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise
        return json.loads(m.group(0))


def _source_tokens(grid: List[List[str]]) -> set[str]:
    toks: set[str] = set()
    for row in grid:
        for cell in row:
            t = normalize_ws(cell)
            if t:
                toks.add(t.lower())
                for w in re.findall(r"[≥≤<>]?\d+(?:[.,]\d+)?%?", t.lower()):
                    toks.add(w)
    return toks


def _looks_invented(value: str, source: set[str]) -> bool:
    """Reject numeric tokens that never appear in the source grid."""
    v = normalize_ws(value)
    if not v:
        return False
    # Allow status phrases / empty
    if re.search(
        r"(?i)\b(insufficient|not\s+applicable|not\s+reported|n/?a|"
        r"illegible|redacted|uncertain)\b",
        v,
    ):
        return False
    nums = re.findall(r"[≥≤<>]?\d+(?:[.,]\d+)?%?", v.lower())
    if not nums:
        return False
    return any(n not in source and n.rstrip("%") not in source for n in nums)


def validate_and_merge_llm_result(
    table: ExtractedTable,
    llm: Dict[str, Any],
    *,
    original_grid: List[List[str]],
) -> Tuple[ExtractedTable, Dict[str, Any]]:
    """Apply LLM output with source-fidelity checks. Mutates a copy-like update."""
    usage: Dict[str, Any] = {"applied": False, "rejected_cells": 0, "warnings": []}
    source = _source_tokens(original_grid)
    headers = llm.get("reconstructed_headers") or llm.get("original_headers")
    rows = llm.get("rows")
    if not isinstance(headers, list) or not isinstance(rows, list) or not rows:
        usage["warnings"].append("llm_result_missing_headers_or_rows")
        table.warnings.append("bedrock:rejected_invalid_shape")
        return table, usage

    width = max(len(headers), max((len(r) for r in rows if isinstance(r, list)), default=0), 1)
    headers = [normalize_ws(str(h)) for h in pad_row([str(x) for x in headers], width)]
    clean_rows: List[List[str]] = []
    statuses = llm.get("cell_statuses")
    rejected = 0
    for ri, row in enumerate(rows):
        if not isinstance(row, list):
            usage["warnings"].append(f"llm_bad_row_{ri}")
            continue
        out_row = pad_row([normalize_ws(str(c)) for c in row], width)
        for ci, cell in enumerate(out_row):
            if _looks_invented(cell, source):
                # Fall back to structural cell when possible
                if ri < len(table.rows) and ci < len(table.rows[ri]):
                    out_row[ci] = table.rows[ri][ci]
                else:
                    out_row[ci] = ""
                rejected += 1
        clean_rows.append(out_row)

    if not clean_rows:
        usage["warnings"].append("llm_all_rows_rejected")
        table.warnings.append("bedrock:rejected_empty_body")
        return table, usage

    # Header fidelity: paths may join existing tokens; reject headers with new numbers
    safe_headers: List[str] = []
    for i, h in enumerate(headers):
        if _looks_invented(h, source) and i < len(table.headers):
            safe_headers.append(table.headers[i])
            rejected += 1
        else:
            safe_headers.append(h or (table.headers[i] if i < len(table.headers) else f"Column {i+1}"))

    table.headers = safe_headers
    table.rows = clean_rows
    if isinstance(llm.get("table_title"), str) and llm["table_title"].strip():
        table.caption = normalize_ws(llm["table_title"])
    if isinstance(llm.get("footnotes"), list):
        table.footnotes = [normalize_ws(str(f)) for f in llm["footnotes"] if normalize_ws(str(f))]
    for w in llm.get("reconstruction_warnings") or []:
        table.warnings.append(f"bedrock:{normalize_ws(str(w))}")
    for u in llm.get("unresolved") or []:
        table.warnings.append(f"unresolved:{normalize_ws(str(u))}")
    conf = llm.get("extraction_confidence")
    if isinstance(conf, (int, float)) and 0 <= float(conf) <= 1:
        table.confidence = float(conf)

    # Bedrock often leaves vertically-merged categorical labels empty mid-span.
    # Re-fill with structural nearest-block assignment so shared attributes are
    # not lost downstream.
    from ste.shared_attribute_expander import (
        body_row_positions_from_cells_meta,
        expand_marker_regions,
        expand_shared_attributes,
    )

    prov = table.provenance if isinstance(table.provenance, dict) else {}
    cells_meta = prov.get("cells_meta") if isinstance(prov, dict) else None
    header_n = int(prov.get("header_row_count") or 0) if isinstance(prov, dict) else 0
    row_positions = body_row_positions_from_cells_meta(
        cells_meta if isinstance(cells_meta, list) else None,
        header_row_count=header_n,
        body_rows=len(table.rows),
    )
    post_fill = expand_shared_attributes(table.rows, row_positions=row_positions)
    post_fill.extend(expand_marker_regions(table.rows))
    for w in post_fill:
        table.warnings.append(f"bedrock_post:{w}")
        usage["warnings"].append(f"bedrock_post:{w}")

    # Rebuild cells from merged grid
    from ste.cell_status import build_cells

    # Prefer LLM statuses when shape matches
    if (
        isinstance(statuses, list)
        and len(statuses) == len(clean_rows)
        and all(isinstance(r, list) and len(pad_row([str(c) for c in r], width)) == width for r in statuses)
    ):
        # Apply status labels by rewriting marker text only when empty→status name
        for ri, srow in enumerate(statuses):
            srow = pad_row([normalize_ws(str(c)) for c in srow], width)
            for ci, st in enumerate(srow):
                st_l = st.lower()
                if st_l in {
                    "insufficient_data",
                    "not_applicable",
                    "not_reported",
                    "illegible",
                    "redacted",
                    "uncertain",
                } and not normalize_ws(clean_rows[ri][ci]):
                    # Keep empty; status resolved in build_cells / classify
                    pass
                elif st_l == "insufficient_data" and "insufficient" not in clean_rows[ri][ci].lower():
                    # Don't invent; leave value as-is
                    pass

    table.cells = build_cells(table.headers, table.rows, confidence=table.confidence * 100 if table.confidence <= 1 else table.confidence)
    table.provenance = dict(table.provenance or {})
    table.provenance["bedrock_applied"] = True
    table.provenance["reconstructed_headers"] = list(table.headers)
    usage["applied"] = True
    usage["rejected_cells"] = rejected
    if rejected:
        table.warnings.append(f"bedrock:source_fidelity_rejections:{rejected}")
    return table, usage


def call_bedrock_reconstruction(
    *,
    table: ExtractedTable,
    image_bytes: bytes,
    image_format: str = "png",
    document_id: str = "",
    model_id: Optional[str] = None,
    region: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Call Bedrock Converse with page image + structural payload."""
    model_id = model_id or DEFAULT_MODEL_ID
    prov = dict(table.provenance or {})
    original_grid = prov.get("original_grid") or (
        list(table.header_rows or []) + [list(r) for r in table.rows]
    )
    user_text = build_user_payload(
        document_id=document_id or str(prov.get("document_id") or ""),
        page_number=table.page,
        table_id=table.table_id,
        caption=table.caption,
        original_headers=list(prov.get("original_headers") or table.headers),
        reconstructed_headers=list(table.headers),
        header_rows=[list(r) for r in table.header_rows],
        body_rows=[list(r) for r in table.rows],
        original_grid=original_grid,
        bbox=table.bbox.as_list(),
        extraction_method=table.extraction_method,
        structural_warnings=list(table.warnings),
    )

    client = get_bedrock_client(region=region)
    fmt = "jpeg" if image_format.lower() in {"jpg", "jpeg"} else "png"
    content: List[Dict[str, Any]] = [
        {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
        {"text": user_text},
    ]

    resp = client.converse(
        modelId=model_id,
        system=[{"text": UNIVERSAL_SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": 8192, "temperature": 0.0},
    )
    usage = {
        "model_id": model_id,
        "input_tokens": (resp.get("usage") or {}).get("inputTokens"),
        "output_tokens": (resp.get("usage") or {}).get("outputTokens"),
        "total_tokens": (resp.get("usage") or {}).get("totalTokens"),
    }
    parts = (((resp.get("output") or {}).get("message") or {}).get("content")) or []
    text = ""
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            text += p["text"]
    parsed = _extract_json(text)
    return parsed, usage


def should_skip_bedrock_simple(table: ExtractedTable) -> Tuple[bool, str]:
    """Return (skip, reason) when Textract grid already looks complete.

    Conservative: only skip small/medium high-confidence rectangular grids with
    few empties. Complex multi-header / sparse tables still go to Bedrock.
    """
    conf = float(getattr(table, "confidence", 0.0) or 0.0)
    if conf > 1.0:
        conf = conf / 100.0
    rows = list(getattr(table, "rows", None) or [])
    headers = list(getattr(table, "headers", None) or [])
    n_rows = len(rows)
    n_cols = max(len(headers), max((len(r) for r in rows), default=0))
    if n_rows < 1 or n_cols < 2:
        return False, "too_small"
    if n_rows > 40 or n_cols > 12:
        return False, "large_grid"
    if conf < 0.88:
        return False, "low_confidence"
    cells = 0
    empty = 0
    for row in rows:
        for cell in row:
            cells += 1
            if not str(cell or "").strip():
                empty += 1
    if cells and (empty / cells) > 0.22:
        return False, "sparse_grid"
    header_rows = list(getattr(table, "header_rows", None) or [])
    # Multi-band headers (incl. false "header + first data row") need Bedrock.
    if len(header_rows) >= 2:
        return False, "multi_header"
    if any(" / " in str(h or "") for h in headers):
        return False, "joined_header_paths"
    return True, "simple_high_confidence"


def apply_bedrock_reconstruction(
    table: ExtractedTable,
    *,
    image_bytes: bytes,
    image_format: str = "png",
    document_id: str = "",
    model_id: Optional[str] = None,
    region: Optional[str] = None,
    enabled: bool = True,
    skip_simple: bool = False,
) -> Tuple[ExtractedTable, Dict[str, Any]]:
    """Run Bedrock semantic reconstruction; on failure keep structural table."""
    meta: Dict[str, Any] = {"enabled": enabled, "applied": False}
    if not enabled:
        meta["skipped"] = "disabled"
        return table, meta
    if skip_simple:
        skip, reason = should_skip_bedrock_simple(table)
        if skip:
            table.warnings.append(f"bedrock:skipped_simple:{reason}")
            meta["skipped"] = f"simple:{reason}"
            return table, meta
    if not image_bytes:
        table.warnings.append("bedrock:skipped_no_image")
        meta["skipped"] = "no_image"
        return table, meta
    try:
        llm, usage = call_bedrock_reconstruction(
            table=table,
            image_bytes=image_bytes,
            image_format=image_format,
            document_id=document_id,
            model_id=model_id,
            region=region,
        )
        meta.update(usage)
        if not llm:
            table.warnings.append("bedrock:empty_response")
            return table, meta
        original_grid = (table.provenance or {}).get("original_grid") or []
        table, merge_meta = validate_and_merge_llm_result(
            table, llm, original_grid=original_grid
        )
        meta.update(merge_meta)
        return table, meta
    except Exception as exc:
        logger.warning("Bedrock reconstruction failed: %s", exc)
        table.warnings.append(f"bedrock:error:{type(exc).__name__}:{exc}")
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return table, meta
