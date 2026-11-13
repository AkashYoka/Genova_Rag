"""Diagnose Titan ValidationException during invoice reindex."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from app.config import get_settings
    from app.rag.embeddings.bedrock_embeddings import BedrockEmbeddings
    from app.rag.ingestion.pdf_layout.parser import parse_pdf
    from app.rag.llm.bedrock_client import _aws_error_fields, create_bedrock_runtime_client
    from app.rag.table_rag import expand_table_rag_chunks

    settings = get_settings()
    print(
        "flags",
        settings.PDF_TEXTRACT_FALLBACK_ENABLED,
        settings.PDF_DOCLING_TABLEFORMER_MODE,
        settings.PDF_DOCLING_PAGE_BATCH_SIZE,
    )
    pdf = ROOT / "data/uploads/9b9b6b9335744d2db129ad09bc0090cc.pdf"
    doc_id = "0479d15c-28ba-44c0-a4bc-752bbdc9cdde"
    parsed = parse_pdf(
        str(pdf),
        document_id=doc_id,
        document_name="01_Vendor_Invoice_Goods_CGST_SGST.pdf",
    )
    rep = parsed.report.get("table_extraction") or {}
    out = ROOT / "data/layout_reports/invoice_docling_extract_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "document": "01_Vendor_Invoice_Goods_CGST_SGST.pdf",
        "table_extraction": {
            "engine": rep.get("engine"),
            "engine_info": rep.get("engine_info"),
            "table_count": rep.get("table_count"),
            "pages_processed": rep.get("pages_processed"),
            "pages_with_tables": rep.get("pages_with_tables"),
            "pages_without_tables": rep.get("pages_without_tables"),
            "failure_count": rep.get("failure_count"),
            "failures": rep.get("failures"),
            "tables": rep.get("tables"),
        },
        "page_metrics": [
            {
                "page": p.page_number,
                "metrics": dict(p.metrics or {}),
                "warnings": list(p.warnings or []),
                "n_tables": len(p.tables or []),
            }
            for p in (parsed.pages or [])
        ],
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)

    chunks, _tables = expand_table_rag_chunks(
        list(parsed.chunks or []),
        document_id=doc_id,
        filename="01_Vendor_Invoice_Goods_CGST_SGST.pdf",
        enabled=True,
    )
    print("expanded_chunks", len(chunks))
    emb = BedrockEmbeddings()
    client = create_bedrock_runtime_client(emb.region)
    for i, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "")
        print(f"--- chunk {i} type={chunk.get('chunk_type')} len={len(text)} ---")
        try:
            emb.embed_documents([text])
            print("embed ok")
        except Exception as exc:
            print("embed FAIL", exc)
            try:
                client.invoke_model(
                    modelId=emb.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(
                        {
                            "inputText": text or " ",
                            "dimensions": emb.dimensions,
                            "normalize": True,
                        }
                    ),
                )
            except Exception as exc2:
                code, status, msg = _aws_error_fields(exc2)
                print("aws", code, status, msg)
                print("preview", repr(text[:400]))
                print("has_null", "\x00" in text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
