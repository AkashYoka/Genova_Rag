"""Compare legacy/layout chunking against hierarchical semantic chunking.

Usage (from ``backend/``):

    python scripts/compare_chunking.py --pdf path/to/document.pdf
    python scripts/compare_chunking.py --pdf doc.pdf --strategy deterministic
    python scripts/compare_chunking.py --pdf doc.pdf --json-out comparison.json

The semantic strategy calls Amazon Nova 2 Lite through Bedrock unless
``--strategy deterministic`` is used. The baseline is the layout-aware chunker
that ingestion used before the semantic pipeline was introduced.

The report shows section/chunk counts, token distribution, page coverage,
table preservation, and whether every extracted source block survived into at
least one chunk.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

_WS_RE = re.compile(r"\s+")
_HEADING_MARKER_RE = re.compile(r"^#{1,6}\s+", flags=re.M)


def _normalize(text: str) -> str:
    text = _HEADING_MARKER_RE.sub("", text or "")
    return _WS_RE.sub(" ", text).strip()


def _page_span(chunk: Dict[str, Any]) -> tuple[int, int]:
    start = chunk.get("page_start") or chunk.get("page") or 1
    end = chunk.get("page_end") or start
    try:
        return int(start), int(end)
    except Exception:
        return 1, 1


def _chunk_stats(
    chunks: Sequence[Dict[str, Any]],
    *,
    page_count: int = 0,
    max_tokens: int = 1800,
    token_counter=None,
) -> Dict[str, Any]:
    from app.rag.chunking.semantic.types import estimate_tokens

    counter = token_counter or estimate_tokens
    tokens: List[int] = []
    pages: set[int] = set()
    table_ids: set[str] = set()
    chunk_types: Dict[str, int] = {}
    for chunk in chunks:
        text = chunk.get("text") or chunk.get("content") or ""
        tokens.append(int(counter(text)))
        start, end = _page_span(chunk)
        pages.update(range(start, end + 1))
        chunk_type = str(chunk.get("chunk_type") or "text")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        table_id = chunk.get("table_id")
        if table_id:
            table_ids.add(str(table_id))
        elif chunk.get("table_json"):
            try:
                payload = json.loads(chunk["table_json"])
                if payload.get("table_id"):
                    table_ids.add(str(payload["table_id"]))
            except Exception:
                pass

    ordered = sorted(tokens)
    if ordered:
        p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        token_summary = {
            "min": ordered[0],
            "median": int(statistics.median(ordered)),
            "p95": ordered[p95_index],
            "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 1),
            "total": sum(ordered),
        }
    else:
        token_summary = {
            "min": 0,
            "median": 0,
            "p95": 0,
            "max": 0,
            "mean": 0.0,
            "total": 0,
        }

    expected_pages = set(range(1, page_count + 1)) if page_count else set()
    return {
        "chunk_count": len(chunks),
        "chunk_types": chunk_types,
        "tokens": token_summary,
        "chunks_over_limit": sum(1 for value in tokens if value > max_tokens),
        "page_count": page_count,
        "pages_covered": len(pages),
        "pages_missing": sorted(expected_pages - pages),
        "table_ids": sorted(table_ids),
        "table_count": len(table_ids),
    }


def _content_preservation(
    blocks: Sequence[Any],
    chunks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check that every extracted block still appears in at least one chunk."""
    if not blocks:
        return {
            "block_count": 0,
            "preserved": 0,
            "missing_block_ids": [],
            "preserved_ratio": 1.0,
            "tables_expected": 0,
            "tables_preserved": 0,
            "missing_table_ids": [],
        }

    combined = _normalize("\n".join(chunk.get("text") or "" for chunk in chunks))
    table_json = "\n".join(chunk.get("table_json") or "" for chunk in chunks)
    referenced: set[str] = set()
    for chunk in chunks:
        raw = chunk.get("source_block_ids")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = [raw]
        if isinstance(raw, list):
            referenced.update(str(item) for item in raw)

    missing: List[str] = []
    tables_expected: set[str] = set()
    tables_preserved: set[str] = set()
    for block in blocks:
        if block.is_table:
            table_id = str((block.table or {}).get("table_id") or "")
            if table_id:
                tables_expected.add(table_id)
                if table_id in table_json:
                    tables_preserved.add(table_id)
        snippet = _normalize(block.text or "")[:120]
        if not snippet:
            continue
        if block.block_id in referenced or snippet in combined:
            continue
        # Splitting may break a long paragraph at a sentence boundary.
        shorter = snippet[:60]
        if shorter and shorter in combined:
            continue
        missing.append(block.block_id)

    missing_tables = sorted(tables_expected - tables_preserved)
    preserved = len(blocks) - len(missing)
    return {
        "block_count": len(blocks),
        "preserved": preserved,
        "missing_block_ids": missing[:50],
        "missing_block_count": len(missing),
        "preserved_ratio": round(preserved / len(blocks), 4),
        "tables_expected": len(tables_expected),
        "tables_preserved": len(tables_preserved),
        "missing_table_ids": missing_tables,
    }


def _run_baseline(
    pdf: Path,
    *,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> List[Dict[str, Any]]:
    from app.services.layout_parser import AdvancedPDFParser

    return AdvancedPDFParser(str(pdf)).chunk_document(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _run_semantic(
    pdf: Path,
    *,
    document_id: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
):
    from app.rag.chunking.semantic.service import semantic_chunk_document

    return semantic_chunk_document(
        str(pdf),
        document_id=document_id,
        document_name=pdf.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _print_table(report: Dict[str, Any]) -> None:
    baseline = report["baseline"]
    semantic = report["semantic"]
    rows = [
        ("chunks", baseline["chunk_count"], semantic["chunk_count"]),
        ("sections", "n/a", semantic.get("section_count", 0)),
        ("tokens median", baseline["tokens"]["median"], semantic["tokens"]["median"]),
        ("tokens p95", baseline["tokens"]["p95"], semantic["tokens"]["p95"]),
        ("tokens max", baseline["tokens"]["max"], semantic["tokens"]["max"]),
        (
            "over token limit",
            baseline["chunks_over_limit"],
            semantic["chunks_over_limit"],
        ),
        ("pages covered", baseline["pages_covered"], semantic["pages_covered"]),
        ("tables", baseline["table_count"], semantic["table_count"]),
        (
            "blocks preserved",
            "n/a",
            f"{semantic['content']['preserved']}/{semantic['content']['block_count']}",
        ),
    ]
    print(f"\nChunking comparison for: {report['pdf']}")
    print(f"strategy: {report['strategy']}\n")
    print(f"{'metric':<20} {'baseline':>12} {'semantic':>12}")
    print(f"{'-' * 20} {'-' * 12} {'-' * 12}")
    for label, left, right in rows:
        print(f"{label:<20} {str(left):>12} {str(right):>12}")

    missing = semantic["content"].get("missing_block_ids") or []
    if missing:
        print(f"\nWARNING: {len(missing)} source block(s) not found in chunks: {missing[:10]}")
    if semantic["pages_missing"]:
        print(f"WARNING: pages with no chunk: {semantic['pages_missing'][:20]}")
    if semantic.get("report", {}).get("used_fallback"):
        print("NOTE: semantic run used the deterministic fallback.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Path to a PDF file")
    parser.add_argument(
        "--strategy",
        choices=("semantic_llm", "deterministic"),
        default="semantic_llm",
        help="Semantic strategy to compare (default: semantic_llm)",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=None,
        help="Override SEMANTIC_CHUNKING_MAX_CHUNK_TOKENS",
    )
    parser.add_argument("--doc-id", default="compare-doc")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--json-out", default=None, help="Write the full report as JSON")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"PDF not found: {pdf}", file=sys.stderr)
        return 1

    if args.strategy == "deterministic":
        os.environ["CHUNK_STRATEGY"] = "deterministic"
    if args.max_chunk_tokens:
        os.environ["SEMANTIC_CHUNKING_MAX_CHUNK_TOKENS"] = str(args.max_chunk_tokens)

    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    max_tokens = int(
        getattr(settings, "SEMANTIC_CHUNKING_MAX_CHUNK_TOKENS", 1800) or 1800
    )

    baseline_chunks: List[Dict[str, Any]] = []
    if not args.skip_baseline:
        try:
            baseline_chunks = _run_baseline(
                pdf,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        except Exception as exc:  # diagnostic tool: keep going
            print(f"Baseline chunking failed: {exc}", file=sys.stderr)

    result = _run_semantic(
        pdf,
        document_id=args.doc_id,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    semantic_chunks = list(result)
    blocks = getattr(result, "source_blocks", None) or []
    page_count = int(getattr(result, "page_count", 0) or 0)

    report: Dict[str, Any] = {
        "pdf": str(pdf),
        "strategy": getattr(result, "strategy", args.strategy),
        "index_version": getattr(result, "index_version", ""),
        "baseline": _chunk_stats(
            baseline_chunks,
            page_count=page_count,
            max_tokens=max_tokens,
        ),
        "semantic": {
            **_chunk_stats(
                semantic_chunks,
                page_count=page_count,
                max_tokens=max_tokens,
            ),
            "section_count": len(getattr(result, "hierarchy", []) or []),
            "root_sections": len(
                [
                    node
                    for node in (getattr(result, "hierarchy", []) or [])
                    if node.parent_section_id is None
                ]
            ),
            "content": _content_preservation(blocks, semantic_chunks),
            "report": getattr(result, "report", {}) or {},
        },
    }

    _print_table(report)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nJSON report written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
