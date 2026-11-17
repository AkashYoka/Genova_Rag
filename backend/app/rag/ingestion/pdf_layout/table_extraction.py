"""Unified document table extraction for the PDF layout stage.

Coordinates STE (AnalyzeDocument TABLES + Bedrock) and Textract
DetectDocumentText into one plan, then resolves per-page tables with a
single merge policy (STE wins overlaps).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.rag.ingestion.pdf_layout.types import TableData

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int], None]


@dataclass
class TableExtractionPlan:
    """Document-level table extraction result used by ``parse_pdf``."""

    engine: str = "none"
    ste_enabled: bool = False
    ste_tables: List[TableData] = field(default_factory=list)
    ste_meta: Dict[str, Any] = field(default_factory=dict)
    ste_error: str = ""
    textract_enabled: bool = False
    textract_tables_by_page: Dict[int, List[TableData]] = field(default_factory=dict)
    textract_words_by_page: Dict[int, List[Any]] = field(default_factory=dict)
    textract_meta: Dict[str, Any] = field(default_factory=dict)
    textract_error: str = ""
    engine_info: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_report(self) -> Dict[str, Any]:
        info = dict(self.engine_info or {})
        return {
            "engine": self.engine,
            "ste_enabled": self.ste_enabled,
            "ste_table_count": len(self.ste_tables),
            "ste_error": self.ste_error or None,
            "textract_enabled": self.textract_enabled,
            "textract_table_count": sum(
                len(v) for v in self.textract_tables_by_page.values()
            ),
            "textract_text_pages": len(self.textract_words_by_page),
            "textract_error": self.textract_error or None,
            "elapsed_s": self.elapsed_s,
            "detect_skipped": bool(info.get("detect_skipped")),
            "detect_elapsed_s": info.get("detect_elapsed_s"),
            "ste_elapsed_s": (info.get("ste") or {}).get("elapsed_s"),
            "engine_info": info,
        }


def _textract_allowed(settings: Any) -> bool:
    return bool(getattr(settings, "PDF_TEXTRACT_FALLBACK_ENABLED", False)) and bool(
        getattr(settings, "PDF_TEXTRACT_TABLES_ENABLED", True)
    )


def _ste_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "PDF_STE_TABLES_ENABLED", False))


def resolve_table_engine(settings: Any) -> str:
    if _ste_enabled(settings):
        return "ste_textract_analyze_bedrock"
    if _textract_allowed(settings):
        return "textract_detect"
    return "none"


def run_document_table_extraction(
    filepath: str,
    *,
    page_sizes: Optional[Dict[int, Tuple[float, float]]] = None,
    doc: Any = None,
    document_id: str = "",
    document_name: str = "",
    settings: Any = None,
    progress: Optional[ProgressCallback] = None,
) -> TableExtractionPlan:
    """Run document-level STE + DetectDocumentText as one extraction plan."""
    from app.config import get_settings

    if settings is None:
        settings = get_settings()

    started = time.perf_counter()
    path = Path(filepath)
    engine = resolve_table_engine(settings)
    ste_on = _ste_enabled(settings)
    textract_on = _textract_allowed(settings)

    plan = TableExtractionPlan(
        engine=engine,
        ste_enabled=ste_on,
        textract_enabled=textract_on,
    )
    plan.engine_info = {
        "engine": engine,
        "ste_tables_enabled": ste_on,
        "ste_bedrock_enabled": bool(getattr(settings, "PDF_STE_BEDROCK_ENABLED", True)),
        "ste_tables_primary": ste_on,
        "textract_fallback_enabled": textract_on,
        "textract_primary": textract_on,
        "pdf_textract_fallback_enabled": bool(
            getattr(settings, "PDF_TEXTRACT_FALLBACK_ENABLED", False)
        ),
        "pdf_textract_tables_enabled": bool(
            getattr(settings, "PDF_TEXTRACT_TABLES_ENABLED", True)
        ),
        "textract_async_multipage": False,
        "textract_input": "page_images",
        "tabular_only": bool(getattr(settings, "PDF_TEXTRACT_TABULAR_ONLY", True)),
        "analyze_document_tables": ste_on,
        "region": (
            (getattr(settings, "PDF_TEXTRACT_REGION", "") or "").strip()
            or (getattr(settings, "BEDROCK_REGION", "") or "").strip()
            or "us-east-1"
        ),
        # Legacy keys kept for report compatibility.
        "docling_enabled": False,
        "docling_tableformer_mode": "accurate",
    }

    logger.info(
        "TableExtractionPlan start file=%s engine=%s ste=%s textract=%s",
        path.name,
        engine,
        ste_on,
        textract_on,
    )

    # ── Stage A: STE AnalyzeDocument + Bedrock ─────────────────────────────
    if ste_on:
        try:
            from app.rag.ingestion.pdf_layout.ste_tables import (
                extract_ste_tables_from_pdf,
            )

            def _ste_progress(stage: str, pct: int) -> None:
                if progress:
                    try:
                        # Nest STE progress inside layout_extract band (~12–40).
                        mapped = 12 + int(max(0, min(100, pct)) * 0.18)
                        progress(f"tables_ste:{stage}", mapped)
                    except Exception:
                        pass

            plan.ste_tables, plan.ste_meta = extract_ste_tables_from_pdf(
                str(path),
                document_id=document_id or "",
                document_name=document_name or path.name,
                progress=_ste_progress,
                settings=settings,
            )
            plan.engine_info["ste"] = dict(plan.ste_meta or {})
            logger.info(
                "TableExtractionPlan STE done tables=%s bedrock_tokens=%s",
                len(plan.ste_tables),
                (plan.ste_meta.get("bedrock_usage") or {}).get("total_tokens"),
            )
        except Exception as exc:
            plan.ste_error = f"{type(exc).__name__}: {exc}"
            plan.engine_info["ste_error"] = plan.ste_error
            logger.error("STE table extraction failed for %s: %s", path.name, exc)

    # ── Stage B: DetectDocumentText (prose words + heuristic tables) ───────
    # When STE is primary, Detect is usually redundant (tables from STE;
    # prose from native PDF text). Flag keeps Detect merge for scanned docs.
    skip_detect = bool(
        ste_on
        and getattr(settings, "PDF_STE_SKIP_DETECT_WHEN_STE", True)
        and plan.ste_tables
        and not plan.ste_error
    )
    # If STE failed or returned nothing, keep Detect as safety net when enabled.
    if ste_on and not plan.ste_tables and textract_on:
        skip_detect = False
        plan.engine_info["detect_skip_overridden"] = "ste_empty_or_failed"
    plan.engine_info["skip_detect_when_ste"] = bool(
        getattr(settings, "PDF_STE_SKIP_DETECT_WHEN_STE", True)
    )
    plan.engine_info["detect_skipped"] = skip_detect

    if textract_on and not skip_detect:
        t_detect = time.perf_counter()
        try:
            from app.rag.ingestion.pdf_layout.textract_tables import (
                extract_tables_from_pdf_detect,
                get_textract_api_usage,
            )

            all_textract, textract_meta = extract_tables_from_pdf_detect(
                str(path), page_sizes=page_sizes, doc=doc
            )
            plan.textract_meta = dict(textract_meta or {})
            raw_words = dict(plan.textract_meta.pop("words_by_page", {}) or {})
            for page_no, words in raw_words.items():
                plan.textract_words_by_page[int(page_no)] = list(words or [])
            for table in all_textract:
                plan.textract_tables_by_page.setdefault(int(table.page), []).append(
                    table
                )
            plan.engine_info["textract_doc"] = dict(plan.textract_meta)
            try:
                plan.engine_info["textract_api_usage"] = get_textract_api_usage()
            except Exception:
                pass
            plan.engine_info["textract_text_pages"] = len(plan.textract_words_by_page)
            plan.engine_info["detect_elapsed_s"] = round(
                time.perf_counter() - t_detect, 3
            )
            logger.info(
                "TableExtractionPlan Detect done pages=%s tables=%s elapsed=%ss",
                plan.textract_meta.get("pages_processed"),
                len(all_textract),
                plan.engine_info["detect_elapsed_s"],
            )
        except Exception as exc:
            plan.textract_error = f"{type(exc).__name__}: {exc}"
            plan.engine_info["textract_doc_error"] = plan.textract_error
            plan.engine_info["detect_elapsed_s"] = round(
                time.perf_counter() - t_detect, 3
            )
            logger.error(
                "DetectDocumentText failed for %s: %s", path.name, exc
            )
    elif skip_detect:
        plan.textract_enabled = False
        plan.engine_info["textract_primary"] = False
        plan.engine_info["detect_elapsed_s"] = 0.0
        logger.info(
            "TableExtractionPlan Detect skipped (STE primary, "
            "PDF_STE_SKIP_DETECT_WHEN_STE=true, ste_tables=%s)",
            len(plan.ste_tables),
        )

    plan.elapsed_s = round(time.perf_counter() - started, 3)
    logger.info(
        "TableExtractionPlan finished engine=%s ste_tables=%s detect_tables=%s elapsed=%ss",
        plan.engine,
        len(plan.ste_tables),
        sum(len(v) for v in plan.textract_tables_by_page.values()),
        plan.elapsed_s,
    )
    return plan


def resolve_page_tables(
    page_number: int,
    plan: TableExtractionPlan,
    *,
    page: Any,
    hint_tables: Optional[List[TableData]] = None,
    native_text: str = "",
    settings: Any = None,
    page_looks_tabular_fn: Optional[Callable[..., Tuple[bool, str]]] = None,
) -> Tuple[List[TableData], str, Dict[str, Any], List[str]]:
    """Resolve final tables for one page from the document plan.

    Returns ``(tables, table_source, metrics, warnings)``.
    """
    from app.config import get_settings
    from app.rag.ingestion.pdf_layout.extraction_quality import filter_quality_tables
    from app.rag.ingestion.pdf_layout.textract_tables import (
        extract_tables_from_page,
        usable_textract_tables,
    )

    if settings is None:
        settings = get_settings()

    metrics: Dict[str, Any] = {
        "table_engine": plan.engine,
        "ste_enabled": plan.ste_enabled,
        "textract_primary": plan.textract_enabled,
        "docling_enabled": False,
        "docling_tables_raw": 0,
        "docling_tables": 0,
        "docling_rejected": 0,
    }
    warnings: List[str] = []
    if plan.ste_error and not plan.ste_tables:
        metrics["ste_error"] = plan.ste_error
        warnings.append(f"ste: {plan.ste_error}")
    if plan.textract_error:
        metrics["textract_doc_error"] = plan.textract_error
    if plan.textract_meta:
        metrics["textract_doc_api"] = plan.textract_meta.get("api")

    tabular_only = bool(getattr(settings, "PDF_TEXTRACT_TABULAR_ONLY", True))
    looks_tabular = True
    tabular_reason = "not_checked"
    if page_looks_tabular_fn is not None:
        looks_tabular, tabular_reason = page_looks_tabular_fn(
            page,
            fallback_tables=hint_tables or [],
            page_text=native_text,
        )
    metrics["tabular_candidate"] = looks_tabular
    metrics["tabular_reason"] = tabular_reason

    final_tables: List[TableData] = []
    table_source = "none"
    textract_tables: List[TableData] = []
    textract_error = ""
    textract_started = time.perf_counter()
    run_textract = False
    need_fallback = False

    def _apply_textract(raw_tables: List[TableData], *, source_label: str) -> None:
        nonlocal final_tables, table_source
        usable = usable_textract_tables(raw_tables)
        fb_accepted, _fb_rej, fb_reports = filter_quality_tables(usable)
        metrics["textract_quality"] = fb_reports
        metrics["textract_tables"] = len(usable)
        if fb_accepted:
            final_tables = list(fb_accepted)
            table_source = source_label
            return
        if usable:
            usable_uncertain = []
            for t in usable:
                reasons = list(
                    ((t.provenance or {}).get("quality") or {}).get("reasons") or []
                )
                if any(r in reasons for r in ("too_few_rows", "too_few_columns")):
                    continue
                usable_uncertain.append(t)
            if usable_uncertain:
                final_tables = list(usable_uncertain)
                table_source = source_label
            else:
                metrics["textract_rejected_empty"] = True

    if plan.textract_enabled:
        skip_non_tabular = bool(tabular_only and not looks_tabular)
        precomputed = list(plan.textract_tables_by_page.get(page_number) or [])
        if skip_non_tabular:
            metrics["textract_skipped"] = "non_tabular_page"
            run_textract = False
            if precomputed:
                try:
                    _apply_textract(precomputed, source_label="textract_detect")
                    metrics["textract_applied_precomputed"] = True
                except Exception as exc:
                    textract_error = f"{type(exc).__name__}: {exc}"
                    warnings.append(f"textract: {textract_error}")
        else:
            run_textract = True
            try:
                if precomputed or not plan.textract_error:
                    textract_tables = precomputed
                    _apply_textract(textract_tables, source_label="textract_detect")
                else:
                    textract_tables = extract_tables_from_page(page, page_number)
                    _apply_textract(textract_tables, source_label="textract_detect")
            except Exception as exc:
                textract_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "DetectDocumentText failed on page %s: %s", page_number, exc
                )
                warnings.append(f"textract: {textract_error}")
        need_fallback = bool(run_textract and not final_tables)

    metrics["textract_elapsed_s"] = round(time.perf_counter() - textract_started, 3)
    metrics["textract_invoked"] = bool(run_textract)
    metrics.setdefault("textract_tables", len(textract_tables))
    if textract_error:
        metrics["textract_error"] = textract_error
    if run_textract and not textract_tables and not textract_error:
        metrics["textract_empty"] = True

    if need_fallback and not final_tables:
        metrics["table_extraction_failed"] = True
        warnings.append(
            "table_extraction_failed: DetectDocumentText produced no usable table"
        )
    elif (
        table_source in {"textract_fallback", "textract_detect"}
        and final_tables
        and any(
            (t.provenance or {}).get("quality", {}).get("ok") is False
            for t in final_tables
        )
    ):
        metrics["table_extraction_uncertain"] = True
        for t in final_tables:
            prov = dict(t.provenance or {})
            prov["extraction_uncertain"] = True
            t.provenance = prov

    # STE merge (primary when enabled).
    if plan.ste_enabled:
        from app.rag.ingestion.pdf_layout.ste_tables import (
            EXTRACTION_ENGINE as _STE_ENGINE,
            merge_tables_prefer_ste,
            ste_tables_for_page,
        )

        page_ste = ste_tables_for_page(plan.ste_tables, page_number)
        iou_thr = float(getattr(settings, "PDF_STE_IOU_DEDUPE", 0.55) or 0.55)
        before_n = len(final_tables)
        final_tables = merge_tables_prefer_ste(
            final_tables, page_ste, iou_threshold=iou_thr
        )
        metrics["ste_tables"] = len(page_ste)
        metrics["ste_deduped_textract"] = max(
            0, before_n + len(page_ste) - len(final_tables)
        )
        if page_ste:
            table_source = _STE_ENGINE

    return final_tables, table_source, metrics, warnings
