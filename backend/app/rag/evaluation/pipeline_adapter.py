"""Run evaluation cases through the live Genova retrieve → interpret → generate path."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def resolve_document_id(
    *,
    user_id: str,
    document_id: Optional[str] = None,
    original_name: Optional[str] = None,
) -> Optional[str]:
    if document_id:
        return str(document_id)
    if not original_name:
        return None
    from app.database import SessionLocal
    from app.models import Document

    db = SessionLocal()
    try:
        needle = original_name.lower()
        rows = (
            db.query(Document)
            .filter(Document.user_id == user_id, Document.is_deleted.is_(False))
            .all()
        )
        for row in rows:
            if needle in str(row.original_name or "").lower():
                return str(row.id)
    finally:
        db.close()
    return None


def _chunk_to_context(chunk: Dict[str, Any]) -> str:
    text = str(chunk.get("text") or "").strip()
    meta_bits = []
    if chunk.get("page") is not None:
        meta_bits.append(f"page={chunk.get('page')}")
    if chunk.get("table_id"):
        meta_bits.append(f"table_id={chunk.get('table_id')}")
    if chunk.get("section"):
        meta_bits.append(f"section={chunk.get('section')}")
    prefix = f"[{', '.join(meta_bits)}] " if meta_bits else ""
    return f"{prefix}{text}".strip()


def run_live_pipeline_case(
    case: Dict[str, Any],
    *,
    user_id: str,
    llm_provider: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute one eval case on the real application RAG pipeline.

    Uses ``retrieve`` + ``interpret_table_question`` + ``generate_answer`` —
    the same path as production chat, not a simplified eval-only retriever.
    """
    from app.rag.generation.table_interpretation import interpret_table_question
    from app.rag.llm.agent import generate_answer
    from app.rag.retrieval.retriever import retrieve
    from app.rag.table_rag.telemetry import (
        begin_table_rag_trace,
        clear_table_rag_trace,
        current_table_rag_trace,
    )

    question = str(
        case.get("user_input") or case.get("question") or ""
    ).strip()
    doc_id = resolve_document_id(
        user_id=user_id,
        document_id=case.get("document_id"),
        original_name=case.get("original_name") or case.get("document"),
    )
    k = int(top_k or case.get("top_k") or 8)

    begin_table_rag_trace(question=question, document_id=str(doc_id or ""))
    t0 = time.perf_counter()
    error: Optional[str] = None
    answer = ""
    sources: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    interpretation: Dict[str, Any] = {}
    verified: Optional[Dict[str, Any]] = None

    try:
        chunks = retrieve(
            query=question,
            user_id=user_id,
            document_id=doc_id,
            top_k=k,
        )
        interpretation = interpret_table_question(question, chunks) or {}
        result = generate_answer(
            question=question,
            user_id=user_id,
            document_id=doc_id,
            top_k=k,
            llm_provider=llm_provider,
            chat_history=[],
        )
        answer = str(result.get("answer") or "")
        sources = list(result.get("sources") or [])
        for src in sources:
            if isinstance(src.get("verified_cell"), dict):
                verified = src["verified_cell"]
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        clear_table_rag_trace()

    selector_status = ""
    for src in sources:
        if src.get("selector_status"):
            selector_status = str(src.get("selector_status"))
            break
    if not selector_status and isinstance(interpretation, dict):
        selector_status = str(
            interpretation.get("status")
            or (interpretation.get("payload") or {}).get("status")
            or ""
        )

    contexts = [_chunk_to_context(c) for c in chunks if c.get("text")]
    reference = case.get("reference")
    if reference is None:
        reference = case.get("expected_value")
    has_reference = reference is not None and str(reference).strip() != ""

    return {
        "id": case.get("id") or question[:48],
        "user_input": question,
        "reference": str(reference).strip() if has_reference else None,
        "has_reference": has_reference,
        "reference_contexts": list(case.get("reference_contexts") or []),
        "response": answer,
        "retrieved_contexts": contexts,
        "sources": sources,
        "chunks_meta": [
            {
                "page": c.get("page"),
                "table_id": c.get("table_id"),
                "section": c.get("section"),
                "score": c.get("score"),
            }
            for c in chunks
        ],
        "verified_cell": verified,
        "selector_status": selector_status,
        "interpretation": {
            "status": interpretation.get("status") if isinstance(interpretation, dict) else None,
            "selected_table_id": (
                interpretation.get("selected_table_id")
                if isinstance(interpretation, dict)
                else None
            ),
        },
        "document_id": doc_id,
        "original_name": case.get("original_name") or case.get("document"),
        "expected_page": case.get("expected_page"),
        "expected_table": case.get("expected_table") or case.get("expected_table_id"),
        "tags": list(case.get("tags") or []),
        "eval_mode": case.get("eval_mode") or (
            "reference" if has_reference else "no_reference"
        ),
        "latency_ms": round(elapsed_ms, 2),
        "error": error,
        "success": error is None,
    }


def load_dataset(path: Path) -> Dict[str, Any]:
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {"name": Path(path).stem, "cases": raw}
    if isinstance(raw, dict) and "cases" in raw:
        return raw
    raise ValueError(f"Dataset must be a list or object with 'cases': {path}")
