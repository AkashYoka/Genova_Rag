"""CLI entry: ``python -m ste --input file.pdf --output out/``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from ste.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone domain-agnostic PDF table extraction and reconstruction. "
            "Primary backend: AWS Textract AnalyzeDocument TABLES on page images; "
            "optional Bedrock LLM reconstruction."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to PDF")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--backend",
        choices=["textract", "pymupdf", "auto"],
        default=None,
        help="Extraction engine (default: textract)",
    )
    parser.add_argument(
        "--gate",
        choices=["cues", "layout"],
        default=None,
        help=(
            "cues: cheap PyMuPDF table-page gate (default); "
            "layout: send every selected page to the backend"
        ),
    )
    parser.add_argument(
        "--ocr",
        choices=["auto", "tesseract", "easyocr", "rapidocr", "none"],
        default=None,
        help="Local OCR engine for pymupdf fallback (default: tesseract)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Render DPI for page images / previews / LLM (default: 200)",
    )
    parser.add_argument(
        "--tables-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Process only pages with a table signal (default: true)",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Comma-separated 1-based page numbers (e.g. 5,6,7,8)",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache backend results for unchanged PDFs (default: true)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: .ste_cache in the project)",
    )
    parser.add_argument(
        "--stitch-multipage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Merge repeated-header page-break continuations (default: true)",
    )
    parser.add_argument(
        "--bedrock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional Bedrock LLM semantic reconstruction (default: env STE_BEDROCK)",
    )
    parser.add_argument(
        "--bedrock-model-id",
        default=None,
        help="Bedrock model id (or STE_BEDROCK_MODEL_ID)",
    )
    parser.add_argument(
        "--aws-region",
        default=None,
        help="AWS region for Textract AnalyzeDocument and optional Bedrock",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List locally available extraction backends and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if "--list-backends" in raw_args:
        from ste.backends import available_backends

        print("Available backends:", ", ".join(available_backends()) or "none")
        return 0

    args = build_parser().parse_args(raw_args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    from ste.pipeline import extract_pdf

    pages = None
    if args.pages:
        pages = [int(x.strip()) for x in str(args.pages).split(",") if x.strip()]

    settings = Settings.from_env().with_overrides(
        backend=args.backend,
        gate=args.gate,
        dpi=args.dpi,
        ocr_engine=args.ocr,
        tables_only=args.tables_only,
        pages=tuple(pages) if pages else None,
        use_cache=args.cache,
        stitch_multipage=args.stitch_multipage,
        bedrock=args.bedrock,
        bedrock_model_id=args.bedrock_model_id,
        aws_region=args.aws_region,
    )

    result = extract_pdf(
        args.input,
        output_dir=args.output,
        settings=settings,
        cache_dir=args.cache_dir,
    )
    usage = (result.meta or {}).get("bedrock_usage") or {}
    metrics = (result.meta or {}).get("metrics") or {}
    cache = (result.meta or {}).get("cache") or {}
    print(
        f"Done: tables={len(result.tables)} "
        f"processed_pages={len(result.processed_pages)} "
        f"skipped_pages={len(result.skipped_pages)} "
        f"backend={result.ocr_backend} "
        f"cache_hit={cache.get('hit', False)} "
        f"wall={metrics.get('wall_seconds', 0)}s "
        f"bedrock_tokens={usage.get('total_tokens', 0)} "
        f"output={Path(args.output).resolve()}"
    )
    if result.errors:
        print(f"Errors: {len(result.errors)}", file=sys.stderr)
        for error in result.errors[:10]:
            print(f"  - {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
