"""Backfill empty DocumentTable.caption from page text near the stored bbox.

Domain-agnostic repair for Docling empties without a full re-extract.
Also injects the caption into llm_understanding applicability_cues.

Usage:
  python scripts/backfill_table_captions.py --document-id <uuid>
  python scripts/backfill_table_captions.py --filename-substr xolair
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.models import Document, DocumentTable
from app.rag.ingestion.pdf_layout.caption_recovery import recover_caption_from_pdf_page
from app.rag.ingestion.pdf_layout.types import BBox


def _bbox(row: DocumentTable) -> BBox | None:
    raw = row.bbox_json
    if not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None
    if isinstance(data, dict):
        try:
            return BBox(
                float(data.get("x0", data.get("l", 0)) or 0),
                float(data.get("y0", data.get("t", 0)) or 0),
                float(data.get("x1", data.get("r", 0)) or 0),
                float(data.get("y1", data.get("b", 0)) or 0),
            )
        except Exception:
            return None
    if isinstance(data, (list, tuple)) and len(data) >= 4:
        return BBox(float(data[0]), float(data[1]), float(data[2]), float(data[3]))
    return None


def _resolve_pdf(doc: Document) -> Path | None:
    for attr in ("file_path", "filepath", "local_path", "stored_path"):
        val = getattr(doc, attr, None)
        if val and Path(val).is_file():
            return Path(val)
    uploads = ROOT / "data" / "uploads"
    filename = (doc.filename or "").strip()
    # Upload objects are often stored under the hashed filename.
    if filename:
        cand = uploads / filename
        if cand.is_file():
            return cand
    # Prefer original_name / known sample copies (e.g. xolair_prescribing.pdf).
    original = (getattr(doc, "original_name", None) or "").strip()
    for name in (original, filename):
        if not name:
            continue
        for folder in (
            uploads,
            ROOT / "data" / "sample_subsets",
            ROOT.parent / "docs",
        ):
            cand = folder / name
            if cand.is_file():
                return cand
            stem = Path(name).stem.lower()
            if folder.is_dir():
                for p in folder.glob("*.pdf"):
                    if stem and stem in p.stem.lower():
                        return p
    # Last resort: if original looks like xolair, use sample subset.
    blob = f"{original} {filename}".lower()
    sample = ROOT / "data" / "sample_subsets" / "xolair_prescribing.pdf"
    if "xolair" in blob and sample.is_file():
        return sample
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--document-id", default="")
    ap.add_argument("--filename-substr", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        q = db.query(Document)
        if args.document_id:
            q = q.filter(Document.id == args.document_id)
        docs = q.all()
        if args.filename_substr:
            sub = args.filename_substr.lower()
            docs = [d for d in docs if sub in (d.filename or "").lower()]
        if not docs:
            print("No documents matched")
            return

        total = 0
        filled = 0
        for doc in docs:
            pdf = _resolve_pdf(doc)
            print(f"doc={doc.id} file={doc.filename!r} pdf={pdf}")
            if not pdf:
                print("  skip: pdf not found")
                continue
            rows = (
                db.query(DocumentTable)
                .filter(DocumentTable.document_id == doc.id)
                .order_by(DocumentTable.page_start, DocumentTable.ordinal)
                .all()
            )
            for row in rows:
                total += 1
                if (row.caption or "").strip():
                    continue
                bbox = _bbox(row)
                if not bbox or not row.page_start:
                    print(f"  p{row.page_start} {row.table_id}: no bbox")
                    continue
                caption = recover_caption_from_pdf_page(
                    str(pdf), int(row.page_start), bbox
                )
                if not caption:
                    print(f"  p{row.page_start} {row.table_id}: no caption found")
                    continue
                print(f"  p{row.page_start} {row.table_id}: {caption[:100]!r}")
                filled += 1
                if args.dry_run:
                    continue
                row.caption = caption
                # Merge into table_json + understanding cues.
                try:
                    tj = json.loads(row.table_json) if row.table_json else {}
                except Exception:
                    tj = {}
                if isinstance(tj, dict):
                    tj["caption"] = caption
                    row.table_json = json.dumps(tj, ensure_ascii=False)
                try:
                    und = (
                        json.loads(row.llm_understanding_json)
                        if row.llm_understanding_json
                        else {}
                    )
                except Exception:
                    und = {}
                if isinstance(und, dict):
                    cues = list(und.get("applicability_cues") or [])
                    if caption not in cues:
                        cues = [caption, *cues]
                    und["applicability_cues"] = cues
                    attrs = list(und.get("applicability_attributes") or [])
                    attrs.insert(
                        0,
                        {
                            "key": "caption",
                            "value": caption,
                            "notes": "",
                            "evidence_type": "explicit",
                            "confidence": 0.9,
                            "provenance": "caption_recovery",
                        },
                    )
                    und["applicability_attributes"] = attrs
                    und["subject"] = caption[:120]
                    # Prefer a less generic purpose when we have a real title.
                    if str(und.get("purpose") or "").startswith("Structured lookup"):
                        und["purpose"] = caption[:200]
                    row.llm_understanding_json = json.dumps(und, ensure_ascii=False)
            if not args.dry_run:
                db.commit()
        print(f"done filled={filled}/{total} dry_run={args.dry_run}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
