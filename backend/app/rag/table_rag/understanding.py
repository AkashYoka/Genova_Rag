"""LLM table understanding during semantic chunking.

Produces interpretation metadata only. Never mutates extracted headers/rows/
cells/merged_cells. Failures fall back to a deterministic skeleton so
ingestion continues.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.rag.table_rag.schema import (
    ApplicabilityAttribute,
    ColumnMeaning,
    HierarchicalHeaderNote,
    LookupAttribute,
    TableUnderstanding,
    TableUnderstandingBatch,
)

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{[\s\S]*\}|\[[\s\S]*\]")

SYSTEM_PROMPT = """You analyze extracted PDF tables for a retrieval system.

CRITICAL RULES:
- Do NOT invent, rewrite, or correct cell values.
- Do NOT invent rows or columns that are not present.
- Never change headers, rows, merged cells, or numeric grid contents.
- Interpret structure only: purpose, header hierarchy, row/column relationships,
  units, footnotes, lookup axes, applicability, and search phrases.
- Use ONLY evidence present in the payload: caption, section_path, surrounding_text,
  headers, header paths, and sample rows.
- If uncertain, set uncertain=true and list uncertainty_notes.
- Return JSON only matching the schema. No markdown fences.

SIBLING TABLES (critical when the batch has more than one table):
- Similar grids often differ by population, category, period, product line, segment,
  grade, indication-class, or section — not by axis units alone.
- For each table, emit applicability_attributes with concrete values copied/paraphrased
  from caption/section/surrounding_text (e.g. key=applicability_class, key=age_band,
  key=reporting_period, key=segment, key=section). Prefer short distinctive values.
- differentiating_dimensions must list the attribute keys that actually separate this
  table from its siblings in THIS document (not generic words like "value" or "dose").
- applicability_cues must include those distinctive phrases so retrieval can tell
  siblings apart. Do not copy another table's class/population onto this one.
- If two tables share axes but apply to different classes, say so in purpose/subject.

Output JSON object:
{
  "tables": [
    {
      "table_id": "...",
      "block_id": "...",
      "description": "one paragraph",
      "purpose": "why this table exists and who/what it applies to",
      "subject": "topic/entity",
      "applicability_cues": [
        "short phrases describing who/what/when this table applies"
      ],
      "applicability_attributes": [
        {"key": "applicability_class|age_band|reporting_period|segment|section|...", "value": "...", "notes": ""}
      ],
      "differentiating_dimensions": [
        "dimensions that distinguish this table from sibling tables"
      ],
      "column_meanings": [
        {"header": "...", "meaning": "...", "unit": "", "is_axis": false, "is_lookup_value": false, "notes": ""}
      ],
      "hierarchical_headers": [
        {"level": 1, "text": "...", "spans_columns": ["..."], "interpretation": "..."}
      ],
      "row_column_relationships": [
        {"description": "...", "related_headers": ["..."]}
      ],
      "units_and_measurements": ["..."],
      "footnotes": [
        {"marker": "*", "explanation": "...", "applies_to": "..."}
      ],
      "semantic_search_phrases": ["..."],
      "suggested_qa_patterns": [
        {"question_pattern": "...", "answer_strategy": "row/col intersection lookup", "required_axes": ["..."]}
      ],
      "lookup_attributes": [
        {"name": "...", "role": "row_axis|col_axis|value|frequency|other", "header_match": "...", "example_values": ["..."], "notes": ""}
      ],
      "numerical_dimensions": ["..."],
      "confidence": 0.0,
      "uncertain": false,
      "uncertainty_notes": []
    }
  ]
}"""


def _extract_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_header(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _header_set(table: Dict[str, Any]) -> set[str]:
    headers = {_normalize_header(h) for h in (table.get("headers") or []) if str(h or "").strip()}
    for row in table.get("header_rows") or []:
        for cell in row or []:
            if str(cell or "").strip():
                headers.add(_normalize_header(cell))
    return headers


def _sample_rows(table: Dict[str, Any], *, max_rows: int) -> List[List[str]]:
    rows = table.get("rows") or []
    out: List[List[str]] = []
    for row in rows[:max_rows]:
        out.append([str(c or "") for c in (row or [])])
    return out


def build_table_prompt_payload(
    table: Dict[str, Any],
    *,
    block_id: str = "",
    section_path: str = "",
    surrounding_text: str = "",
    max_rows: int = 12,
) -> Dict[str, Any]:
    """Compact structured input for the understanding LLM (read-only view)."""
    return {
        "table_id": str(table.get("table_id") or ""),
        "block_id": block_id,
        "page": table.get("page"),
        "section_path": section_path,
        "caption": table.get("caption") or "",
        "extraction_method": table.get("extraction_method") or "",
        "headers": list(table.get("headers") or []),
        "header_rows": [list(r) for r in (table.get("header_rows") or [])][:6],
        "merged_cells": list(table.get("merged_cells") or [])[:24],
        "n_rows": table.get("n_rows"),
        "n_cols": table.get("n_cols"),
        "sample_rows": _sample_rows(table, max_rows=max_rows),
        "low_confidence_cells": list(table.get("low_confidence_cells") or [])[:12],
        "needs_verification": bool(
            table.get("needs_verification") or table.get("low_confidence_cells")
        ),
        "surrounding_context": (surrounding_text or "")[:800],
        "footnote_hints": [
            f for f in (table.get("linearized_facts") or []) if "*" in str(f) or "note" in str(f).lower()
        ][:8],
    }


def validate_understanding_against_table(
    understanding: TableUnderstanding,
    table: Dict[str, Any],
) -> TableUnderstanding:
    """Attach validation flags; never rewrite extracted grid fields."""
    warnings: List[str] = []
    headers = _header_set(table)
    known = headers or set()

    for col in understanding.column_meanings:
        if col.header and known and _normalize_header(col.header) not in known:
            # Allow partial / hierarchical matches.
            if not any(_normalize_header(col.header) in h or h in _normalize_header(col.header) for h in known):
                warnings.append(f"column_meaning header not in table: {col.header!r}")

    for attr in understanding.lookup_attributes:
        match = _normalize_header(attr.header_match or attr.name)
        if match and known and match not in known:
            if not any(match in h or h in match for h in known):
                warnings.append(f"lookup_attribute header_match not in table: {attr.header_match or attr.name!r}")

    if table.get("needs_verification") or table.get("low_confidence_cells"):
        understanding.uncertain = True
        warnings.append("source extraction has low-confidence cells")

    if understanding.confidence < 0.45:
        understanding.uncertain = True

    understanding.validation_warnings = list(
        dict.fromkeys([*(understanding.uncertainty_notes or []), *warnings])
    )
    understanding.validated = True
    understanding.table_id = understanding.table_id or str(table.get("table_id") or "")
    return understanding


def deterministic_understanding(
    table: Dict[str, Any],
    *,
    block_id: str = "",
    section_path: str = "",
) -> TableUnderstanding:
    """No-LLM fallback so indexing still gets structured metadata."""
    headers = [str(h or "") for h in (table.get("headers") or [])]
    caption = str(table.get("caption") or "").strip()
    subject = caption or (section_path.split(">")[-1].strip() if section_path else "table")
    columns = [
        ColumnMeaning(
            header=h,
            meaning=f"Column '{h}' from extracted table",
            is_axis=i == 0,
            is_lookup_value=i > 0 and any(ch.isdigit() for ch in h),
        )
        for i, h in enumerate(headers)
    ]
    attrs: List[LookupAttribute] = []
    if headers:
        attrs.append(
            LookupAttribute(
                name=headers[0],
                role="row_axis",
                header_match=headers[0],
                example_values=[
                    str((table.get("rows") or [[]])[0][0])
                    if table.get("rows")
                    else ""
                ],
            )
        )
        for h in headers[1:]:
            attrs.append(
                LookupAttribute(name=h, role="col_axis", header_match=h)
            )
    cues = [p for p in (caption, section_path) if p]
    app_attrs: List[ApplicabilityAttribute] = []
    if section_path:
        app_attrs.append(
            ApplicabilityAttribute(
                key="section",
                value=section_path,
                evidence_type="explicit",
                confidence=0.9,
                provenance="section",
            )
        )
    # Extract generic numeric age/period intervals from caption/section text.
    for text in list(cues):
        for m in re.finditer(
            r"(?i)(?:ages?\s+(?:of\s+)?)?(?:≥|>=|>|≤|<=|<)?\s*\d+(?:\.\d+)?"
            r"(?:\s*(?:to|–|-|—)\s*(?:≥|>=|>|≤|<=|<)?\s*\d+(?:\.\d+)?)?"
            r"\s*years?(?:\s+of\s+age)?(?:\s+and\s+(?:older|above|over))?"
            r"|\bFY\s?\d{2,4}\b|\bQ[1-4]\s?\d{2,4}\b",
            text,
        ):
            phrase = m.group(0).strip()
            cues.append(phrase)
            key = "reporting_period" if re.search(r"(?i)\bFY\b|\bQ[1-4]\b", phrase) else "age_band"
            if not any(
                a.key == key and a.value.lower() == phrase.lower() for a in app_attrs
            ):
                app_attrs.append(
                    ApplicabilityAttribute(
                        key=key,
                        value=phrase,
                        evidence_type="explicit",
                        confidence=0.85,
                        provenance="caption" if text == caption else "section",
                    )
                )
    # Include reconstructed header paths in search phrases when present.
    paths = table.get("normalized_header_paths") or []
    path_phrases = [
        " > ".join(str(p) for p in path if p)
        for path in paths[:8]
        if isinstance(path, (list, tuple))
    ]
    dims = []
    if any(a.key == "age_band" for a in app_attrs):
        dims.append("age_band")
    if any(a.key == "reporting_period" for a in app_attrs):
        dims.append("reporting_period")
    if cues and not dims:
        dims = ["applicability_class"]
    understanding = TableUnderstanding(
        table_id=str(table.get("table_id") or ""),
        block_id=block_id,
        description=(
            f"Extracted table on page {table.get('page')} with "
            f"{len(table.get('rows') or [])} data rows and {len(headers)} columns"
            + (f" titled '{caption}'" if caption else "")
            + "."
        ),
        purpose="Structured lookup / comparison grid from the source document.",
        subject=subject,
        applicability_cues=list(dict.fromkeys(cues)),
        applicability_attributes=app_attrs,
        differentiating_dimensions=dims,
        column_meanings=columns,
        hierarchical_headers=[
            HierarchicalHeaderNote(
                level=1,
                text=" > ".join(str(p) for p in path),
                spans_columns=[str(path[-1]) if path else ""],
                interpretation="reconstructed_header_path",
            )
            for path in paths[:8]
            if path
        ],
        semantic_search_phrases=[
            p
            for p in [
                caption,
                section_path,
                " ".join(headers[:6]),
                *path_phrases[:4],
                "table lookup schedule matrix",
            ]
            if p
        ],
        lookup_attributes=attrs,
        numerical_dimensions=[h for h in headers if any(ch.isdigit() for ch in h)][:8],
        confidence=0.35,
        uncertain=True,
        uncertainty_notes=["deterministic_fallback_no_llm"],
        source="deterministic_fallback",
    )
    return validate_understanding_against_table(understanding, table)


def _build_user_prompt(payloads: Sequence[Dict[str, Any]], *, document_name: str) -> str:
    n = len(payloads)
    sibling_note = ""
    if n >= 2:
        sibling_note = (
            f"This batch has {n} tables that may be siblings. "
            "Contrast them: emit distinctive applicability_attributes and "
            "differentiating_dimensions for each using caption/section/surrounding_text. "
            "Do not assign the same applicability class to tables that the document "
            "places under different sections or populations.\n"
        )
    return (
        f"Document: {document_name or 'untitled'}\n"
        f"Analyze {n} extracted table(s). "
        "Remember: cell values in sample_rows are ground truth — interpret only; "
        "never rewrite cells.\n"
        f"{sibling_note}\n"
        + json.dumps({"tables": list(payloads)}, ensure_ascii=False, indent=2)
    )


def _parse_batch(raw: Any) -> List[TableUnderstanding]:
    if raw is None:
        return []
    if isinstance(raw, list):
        raw = {"tables": raw}
    if not isinstance(raw, dict):
        return []
    try:
        batch = TableUnderstandingBatch.model_validate(raw)
    except Exception:
        # Tolerate a single-object response.
        try:
            return [TableUnderstanding.model_validate(raw)]
        except Exception:
            return []
    return list(batch.tables)


def understand_tables_llm(
    payloads: Sequence[Dict[str, Any]],
    *,
    document_name: str = "",
    client: Any = None,
    settings: Any = None,
) -> Tuple[List[TableUnderstanding], Dict[str, Any]]:
    """Call the semantic-chunking LLM for one batch of table payloads."""
    settings = settings or get_settings()
    stats = {
        "llm_calls": 0,
        "prompt_chars": 0,
        "elapsed_s": 0.0,
        "used_llm": False,
        "error": "",
    }
    if not payloads:
        return [], stats

    if client is None:
        from app.rag.chunking.semantic.analyzer import build_semantic_client

        client = build_semantic_client(settings)

    prompt = _build_user_prompt(payloads, document_name=document_name)
    stats["prompt_chars"] = len(prompt)
    started = time.perf_counter()
    try:
        stats["llm_calls"] = 1
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=(
                getattr(settings, "SEMANTIC_CHUNKING_MODEL_ID", None)
                or getattr(settings, "BEDROCK_MODEL_ID", None)
                or getattr(settings, "LLM_MODEL", None)
            ),
            max_tokens=int(
                getattr(settings, "TABLE_RAG_LLM_MAX_OUTPUT_TOKENS", 2048) or 2048
            ),
            temperature=0.0,
            stream=False,
        )
        content = ""
        try:
            content = response.choices[0].message.content or ""
        except Exception:
            content = ""
        parsed = _parse_batch(_extract_json(content))
        stats["used_llm"] = bool(parsed)
        stats["elapsed_s"] = round(time.perf_counter() - started, 3)
        return parsed, stats
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        stats["elapsed_s"] = round(time.perf_counter() - started, 3)
        logger.warning("Table understanding LLM call failed: %s", exc)
        return [], stats


def _section_path_for_block(nodes: Sequence[Any], block_id: str) -> str:
    for node in nodes or []:
        if block_id in (getattr(node, "block_ids", None) or []):
            path = getattr(node, "path", None) or []
            return " > ".join(str(p) for p in path if p)
        children = getattr(node, "children", None) or []
        found = _section_path_for_block(children, block_id)
        if found:
            return found
    return ""


def _surrounding_text(
    blocks: Sequence[Any],
    table_block: Any,
    *,
    window: int = 2,
) -> str:
    order = int(getattr(table_block, "order", 0) or 0)
    neighbors = [
        b
        for b in blocks
        if abs(int(getattr(b, "order", 0) or 0) - order) <= window
        and getattr(b, "block_id", None) != getattr(table_block, "block_id", None)
        and not getattr(b, "is_table", False)
    ]
    parts = []
    for b in sorted(neighbors, key=lambda x: int(getattr(x, "order", 0) or 0)):
        text = " ".join(str(getattr(b, "text", "") or "").split())
        if text:
            parts.append(text[:240])
    return " | ".join(parts)


def apply_understanding_to_table(
    table: Dict[str, Any],
    understanding: TableUnderstanding,
) -> Dict[str, Any]:
    """Attach LLM metadata without altering extracted grid fields."""
    # Snapshot original structural fields to prove immutability.
    original_headers = list(table.get("headers") or [])
    original_rows = [list(r) for r in (table.get("rows") or [])]
    original_merged = list(table.get("merged_cells") or [])
    original_cells = list(table.get("cells") or [])
    original_bbox = table.get("bbox")

    enriched = dict(table)
    understanding = validate_understanding_against_table(understanding, table)
    enriched["llm_understanding"] = understanding.to_storage_dict()
    # Never trust LLM to overwrite these.
    enriched["headers"] = original_headers
    enriched["rows"] = original_rows
    enriched["merged_cells"] = original_merged
    enriched["cells"] = original_cells
    enriched["bbox"] = original_bbox
    if understanding.uncertain or understanding.validation_warnings:
        enriched["needs_verification"] = True
        enriched["llm_interpretation_uncertain"] = True
    return enriched


def understand_document_tables(
    blocks: Sequence[Any],
    nodes: Sequence[Any] = (),
    *,
    document_name: str = "",
    client: Any = None,
    settings: Any = None,
    enabled: Optional[bool] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Run table understanding and write metadata onto each table SourceBlock.

    Returns ``(blocks, report)``. Blocks are mutated in place for table payloads.
    """
    settings = settings or get_settings()
    if enabled is None:
        enabled = bool(getattr(settings, "TABLE_RAG_LLM_UNDERSTANDING_ENABLED", True))
        # Also require TableRAG itself.
        enabled = enabled and bool(getattr(settings, "TABLE_RAG_ENABLED", True))

    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "table_count": 0,
        "understood": 0,
        "fallback": 0,
        "llm_calls": 0,
        "prompt_chars": 0,
        "elapsed_s": 0.0,
        "errors": [],
    }
    table_blocks = [b for b in blocks if getattr(b, "is_table", False) and getattr(b, "table", None)]
    report["table_count"] = len(table_blocks)
    if not table_blocks:
        return list(blocks), report

    max_tables = int(getattr(settings, "TABLE_RAG_LLM_MAX_TABLES", 20) or 20)
    batch_size = int(getattr(settings, "TABLE_RAG_LLM_BATCH_SIZE", 2) or 2)
    max_rows = int(getattr(settings, "TABLE_RAG_LLM_PROMPT_ROWS", 12) or 12)
    table_blocks = table_blocks[:max_tables]

    payloads: List[Dict[str, Any]] = []
    meta: List[Tuple[Any, Dict[str, Any], str]] = []
    for block in table_blocks:
        table = dict(block.table or {})
        section_path = _section_path_for_block(nodes, block.block_id) or (
            " > ".join(p for p in (block.section, block.subsection) if p)
        )
        surrounding = _surrounding_text(blocks, block)
        payload = build_table_prompt_payload(
            table,
            block_id=block.block_id,
            section_path=section_path,
            surrounding_text=surrounding,
            max_rows=max_rows,
        )
        payloads.append(payload)
        meta.append((block, table, section_path))

    started = time.perf_counter()
    understandings_by_key: Dict[str, TableUnderstanding] = {}

    if enabled:
        for i in range(0, len(payloads), max(1, batch_size)):
            batch = payloads[i : i + batch_size]
            parsed, stats = understand_tables_llm(
                batch,
                document_name=document_name,
                client=client,
                settings=settings,
            )
            report["llm_calls"] += int(stats.get("llm_calls") or 0)
            report["prompt_chars"] += int(stats.get("prompt_chars") or 0)
            if stats.get("error"):
                report["errors"].append(stats["error"])
            for item in parsed:
                key = (item.table_id or "") + "|" + (item.block_id or "")
                understandings_by_key[key] = item
                if item.table_id:
                    understandings_by_key[item.table_id] = item
                if item.block_id:
                    understandings_by_key[item.block_id] = item

    for block, table, section_path in meta:
        table_id = str(table.get("table_id") or "")
        understanding = (
            understandings_by_key.get(table_id + "|" + block.block_id)
            or understandings_by_key.get(table_id)
            or understandings_by_key.get(block.block_id)
        )
        if understanding is None:
            understanding = deterministic_understanding(
                table, block_id=block.block_id, section_path=section_path
            )
            report["fallback"] += 1
        else:
            understanding.block_id = understanding.block_id or block.block_id
            understanding.table_id = understanding.table_id or table_id
            understanding.source = "llm"
            report["understood"] += 1
        block.table = apply_understanding_to_table(table, understanding)

    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    logger.info(
        "Table understanding: tables=%d llm=%d fallback=%d calls=%d elapsed=%.3fs",
        report["table_count"],
        report["understood"],
        report["fallback"],
        report["llm_calls"],
        report["elapsed_s"],
    )
    return list(blocks), report
