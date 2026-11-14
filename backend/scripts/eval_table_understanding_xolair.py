"""Evaluate LLM table understanding enrichment on xolair_prescribing.pdf.

Compares deterministic-only TableRAG index text vs LLM-enriched index text
for the same extracted tables. Uses a mocked LLM by default so the run is
reproducible offline; pass ``--live-llm`` to call the semantic chunking client.

Examples
--------
  cd backend
  python scripts/eval_table_understanding_xolair.py --offline
  python scripts/eval_table_understanding_xolair.py --live-llm --pages 5,6
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_table_understanding")


def _find_pdf(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"PDF not found: {path}")
        return path
    path = ROOT / "data" / "sample_subsets" / "xolair_prescribing.pdf"
    if not path.exists():
        raise SystemExit("xolair_prescribing.pdf not found")
    return path


def _subset_pdf(src: Path, pages: Sequence[int], dest: Path) -> Dict[int, int]:
    import fitz

    mapping: Dict[int, int] = {}
    with fitz.open(str(src)) as doc:
        out = fitz.open()
        for new_idx, page_no in enumerate(pages, start=1):
            if 1 <= page_no <= doc.page_count:
                out.insert_pdf(doc, from_page=page_no - 1, to_page=page_no - 1)
                mapping[new_idx] = page_no
        out.save(str(dest))
        out.close()
    return mapping


def _mock_understanding_for_table(table: dict) -> dict:
    headers = [str(h or "") for h in (table.get("headers") or [])]
    caption = str(table.get("caption") or "")
    return {
        "table_id": table.get("table_id") or "",
        "description": (
            f"Structured dosing/lookup table{': ' + caption if caption else ''} "
            f"with axes among {', '.join(headers[:4])}."
        ),
        "purpose": "Support exact row/column numerical lookups from prescribing information.",
        "subject": caption or "dosing schedule",
        "column_meanings": [
            {
                "header": h,
                "meaning": f"Extracted column '{h}'",
                "unit": "IU/mL" if "IU" in h else ("kg" if "kg" in h.lower() else ("mg" if "dose" in h.lower() else "")),
                "is_axis": i == 0,
                "is_lookup_value": i > 1,
                "notes": "",
            }
            for i, h in enumerate(headers)
        ],
        "hierarchical_headers": [
            {
                "level": 1,
                "text": str(m.get("text") or ""),
                "spans_columns": [],
                "interpretation": f"Merged cell spanning {m.get('col_span')} columns",
            }
            for m in (table.get("merged_cells") or [])[:4]
            if isinstance(m, dict)
        ],
        "row_column_relationships": [
            {
                "description": "Row axis intersects weight/dose columns for recommended value",
                "related_headers": headers[:3],
            }
        ],
        "units_and_measurements": ["IU/mL", "mg", "kg", "weeks"],
        "footnotes": [],
        "semantic_search_phrases": [
            caption,
            "recommended dose by assay and body weight",
            "dosing table every 2 or 4 weeks",
            " ".join(headers[:5]),
        ],
        "suggested_qa_patterns": [
            {
                "question_pattern": "What dose for assay X IU/mL and weight Y kg?",
                "answer_strategy": "row/col intersection on extracted cells",
                "required_axes": ["assay", "weight"],
            }
        ],
        "lookup_attributes": [
            {
                "name": headers[0] if headers else "row",
                "role": "row_axis",
                "header_match": headers[0] if headers else "",
                "example_values": [
                    str((table.get("rows") or [[""]])[0][0])
                    if table.get("rows")
                    else ""
                ],
                "notes": "",
            }
        ],
        "numerical_dimensions": [h for h in headers if any(ch.isdigit() for ch in h)][:6],
        "confidence": 0.82,
        "uncertain": bool(table.get("needs_verification")),
        "uncertainty_notes": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--pages", default="5,6")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--live-llm", action="store_true")
    parser.add_argument("--json-out", default="data/layout_reports/table_understanding_eval.json")
    parser.add_argument(
        "--query",
        default=(
            "What dose applies for pretreatment assay 350 IU/mL and body weight "
            "78 kg every 2 weeks according to the dosing table?"
        ),
    )
    args = parser.parse_args()

    from app.config import get_settings
    from app.rag.ingestion.pdf_layout.parser import parse_pdf
    from app.rag.table_rag.indexing import expand_table_rag_chunks
    from app.rag.table_rag.representations import table_summary_text
    from app.rag.table_rag.understanding import apply_understanding_to_table
    from app.rag.table_rag.schema import TableUnderstanding as TU
    from app.rag.generation.table_evidence import lookup_cell_in_table
    from app.rag.table_rag.routing import route_query

    settings = get_settings()
    pdf = _find_pdf(args.pdf)
    pages = [int(p) for p in args.pages.split(",") if p.strip()]

    temp_dir = tempfile.TemporaryDirectory(prefix="tu_eval_")
    target = Path(temp_dir.name) / "subset.pdf"
    mapping = _subset_pdf(pdf, pages, target)
    parsed = parse_pdf(str(target), document_name=pdf.name)

    # Collect tables from parse_pdf elements / page tables.
    tables: List[dict] = []
    for page in parsed.pages:
        for table in page.tables or []:
            payload = table.to_json()
            src_page = mapping.get(int(payload.get("page") or page.page_number), page.page_number)
            payload["page"] = src_page
            if not payload.get("table_id"):
                from app.rag.table_rag.ids import make_table_id

                payload["table_id"] = make_table_id(
                    page=src_page,
                    ordinal=len(tables),
                    bbox=payload.get("bbox"),
                    headers=payload.get("headers") or [],
                )
            tables.append(payload)

    before_after = []
    for table in tables:
        before_text = table_summary_text(
            dict(table), section="Dosage", include_llm=True
        )
        if args.live_llm:
            from app.rag.chunking.semantic.types import SourceBlock
            from app.rag.table_rag.understanding import understand_document_tables

            block = SourceBlock(
                block_id=f"b{len(before_after)+1:05d}",
                order=len(before_after),
                element_type="table",
                text=table.get("markdown") or "",
                page_start=int(table.get("page") or 1),
                page_end=int(table.get("page") or 1),
                section="Dosage",
                table=dict(table),
            )
            blocks, report = understand_document_tables(
                [block],
                [],
                document_name=pdf.name,
                settings=settings,
                enabled=True,
            )
            enriched = blocks[0].table
            llm_report = report
        else:
            understanding = TU.model_validate(_mock_understanding_for_table(table))
            enriched = apply_understanding_to_table(dict(table), understanding)
            llm_report = {"mode": "mock", "understood": 1}

        after_text = table_summary_text(
            enriched, section="Dosage", include_llm=True
        )
        # Exact lookup must still use original cells.
        lookup_q = args.query
        before_lookup = lookup_cell_in_table(table, lookup_q, page=table.get("page"))
        after_lookup = lookup_cell_in_table(enriched, lookup_q, page=table.get("page"))
        before_val = getattr(before_lookup, "value", None)
        after_val = getattr(after_lookup, "value", None)

        before_after.append(
            {
                "table_id": table.get("table_id"),
                "page": table.get("page"),
                "caption": table.get("caption"),
                "n_rows": len(table.get("rows") or []),
                "n_cols": table.get("n_cols"),
                "merged_cells": len(table.get("merged_cells") or []),
                "cells_unchanged": enriched.get("rows") == table.get("rows")
                and enriched.get("headers") == table.get("headers")
                and enriched.get("merged_cells") == table.get("merged_cells"),
                "before_summary_chars": len(before_text),
                "after_summary_chars": len(after_text),
                "before_has_llm_block": "LLM TABLE INTERPRETATION" in before_text,
                "after_has_llm_block": "LLM TABLE INTERPRETATION" in after_text,
                "before_summary_head": before_text[:400],
                "after_summary_head": after_text[:600],
                "lookup_before": before_val,
                "lookup_after": after_val,
                "lookup_values_equal": before_val == after_val,
                "llm_report": llm_report,
            }
        )

    route = route_query(args.query)
    # Index expansion before/after for one document id.
    doc_id = str(uuid.uuid4())
    base_chunks = []
    for i, table in enumerate(tables):
        base_chunks.append(
            {
                "text": table.get("markdown") or "table",
                "page": table.get("page"),
                "chunk_index": i,
                "chunk_type": "table",
                "section": "Dosage",
                "table_id": table.get("table_id"),
                "table_json": json.dumps(table),
            }
        )
    expanded_before, _ = expand_table_rag_chunks(
        base_chunks, document_id=doc_id, filename=pdf.name
    )
    enriched_chunks = []
    for i, item in enumerate(before_after):
        table = tables[i]
        # Re-apply the same enrichment used above for fair index compare.
        if item["after_has_llm_block"]:
            # Rebuild from after summary path via mock/live already stored? Use apply again.
            from app.rag.table_rag.schema import TableUnderstanding as TU2
            from app.rag.table_rag.understanding import apply_understanding_to_table as apply2

            enriched = apply2(dict(table), TU2.model_validate(_mock_understanding_for_table(table)))
        else:
            enriched = table
        enriched_chunks.append(
            {
                "text": enriched.get("markdown") or "table",
                "page": enriched.get("page"),
                "chunk_index": i,
                "chunk_type": "table",
                "section": "Dosage",
                "table_id": enriched.get("table_id"),
                "table_json": json.dumps(enriched),
            }
        )
    expanded_after, records = expand_table_rag_chunks(
        enriched_chunks, document_id=doc_id + "-enriched", filename=pdf.name
    )

    result = {
        "pdf": str(pdf),
        "pages": pages,
        "query": args.query,
        "route": {"modality": route.modality.value, "reason": route.reason},
        "tables_evaluated": len(tables),
        "mode": "live_llm" if args.live_llm else "mock_llm",
        "comparisons": before_after,
        "index_chunks_before": len(expanded_before),
        "index_chunks_after": len(expanded_after),
        "summary_index_after_has_interpretation": sum(
            1
            for c in expanded_after
            if c.get("chunk_type") == "table_summary"
            and "LLM TABLE INTERPRETATION" in str(c.get("text") or "")
        ),
        "all_cells_unchanged": all(c["cells_unchanged"] for c in before_after),
        "all_lookups_stable": all(c["lookup_values_equal"] for c in before_after),
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== Table understanding evaluation ===")
    print(f"PDF: {pdf}")
    print(f"Tables: {len(tables)} | mode={result['mode']}")
    print(f"Route: {route.modality.value}")
    print(f"All cells unchanged: {result['all_cells_unchanged']}")
    print(f"Lookup values stable: {result['all_lookups_stable']}")
    for c in before_after:
        print(
            f"- page={c['page']} rows={c['n_rows']} "
            f"summary {c['before_summary_chars']}→{c['after_summary_chars']} chars "
            f"llm_block={c['after_has_llm_block']} lookup_stable={c['lookup_values_equal']}"
        )
    print(f"Wrote {out}")
    temp_dir.cleanup()
    return 0 if result["all_cells_unchanged"] and result["all_lookups_stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
