"""Expand a cellular table into independent row facts for the answer prompt.

Mirrors the assessment-pipeline pattern: each body row is a self-contained
chunk of ``header=value`` pairs so the LLM can match the question against
facts in the user prompt — no domain vocabularies, no product-specific axis
code.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _cell(value: Any) -> str:
    return str(value or "").strip()


def independent_row_facts(
    table: Dict[str, Any],
    *,
    max_rows: Optional[int] = None,
    relevant_row_indices: Optional[Sequence[int]] = None,
) -> List[str]:
    """One prompt line per body row: all headers bound to that row's cells."""
    headers = [_cell(h) for h in (table.get("headers") or [])]
    rows = [list(r) for r in (table.get("rows") or [])]
    if not headers or not rows:
        return []

    if relevant_row_indices:
        keep = sorted(
            {
                int(i)
                for i in relevant_row_indices
                if 0 <= int(i) < len(rows)
            }
        )
        indexed = [(i, rows[i]) for i in keep]
    else:
        indexed = list(enumerate(rows))

    if max_rows is not None and max_rows > 0:
        indexed = indexed[: int(max_rows)]

    facts: List[str] = []
    width = len(headers)
    for ri, row in indexed:
        parts: List[str] = []
        for ci in range(width):
            header = headers[ci] or f"col_{ci}"
            value = _cell(row[ci]) if ci < len(row) else ""
            if not value:
                continue
            parts.append(f"{header}={value}")
        if not parts:
            continue
        facts.append(f"row_fact[r{ri}]: " + " | ".join(parts))
    return facts


def independent_rows_prompt_block(
    table: Dict[str, Any],
    *,
    max_rows: Optional[int] = None,
    relevant_row_indices: Optional[Sequence[int]] = None,
    include_grid: bool = True,
    grid_block: str = "",
) -> str:
    """Assessment-style prompt block: caption + independent row facts (+ optional grid)."""
    caption = _cell(table.get("caption") or table.get("table_title") or "")
    table_id = _cell(table.get("table_id") or "")
    page = table.get("page") or table.get("page_start")
    header_bits = ["INDEPENDENT ROW FACTS"]
    if caption:
        header_bits.append(caption)
    elif table_id:
        header_bits.append(table_id)
    if page not in (None, ""):
        header_bits.append(f"page {page}")

    lines = [
        " — ".join(header_bits),
        "LOOKUP RULE: each row_fact is self-contained. Match the question to "
        "exactly one fact using the header=value pairs; do not mix values "
        "across different row_fact lines.",
    ]
    facts = independent_row_facts(
        table,
        max_rows=max_rows,
        relevant_row_indices=relevant_row_indices,
    )
    if not facts:
        return grid_block.strip()
    lines.extend(facts)
    if include_grid and grid_block.strip():
        lines.append("")
        lines.append(grid_block.strip())
    return "\n".join(lines)
