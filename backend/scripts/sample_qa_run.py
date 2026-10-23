"""Upload a genova-rag sample PDF and ask 12 grounded questions via local API."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:7860/api/v1"
PDF = Path(r"C:\PC\Yoka\Misc\genova-rag\samples\payment\01_Vendor_Invoice_Goods_CGST_SGST.pdf")

QUESTIONS = [
    "What is this document?",
    "Who is the vendor or supplier on this invoice?",
    "Who is the buyer or bill-to party?",
    "What is the invoice number?",
    "What is the invoice date?",
    "What goods or items are listed?",
    "What is the taxable / subtotal amount before tax?",
    "What CGST amount is charged?",
    "What SGST amount is charged?",
    "What is the total invoice amount including tax?",
    "What is the currency used?",
    "Is there a purchase order (PO) number mentioned?",
    "Summarize the payment or tax terms if any are stated.",
    "List all numeric amounts found on the invoice.",
    "What GST rates apply on this invoice?",
]


def main() -> int:
    if not PDF.exists():
        print(f"PDF not found: {PDF}")
        return 1

    with httpx.Client(timeout=300.0) as client:
        # Wait for API
        for i in range(40):
            try:
                r = client.get("http://127.0.0.1:7860/api/health")
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            print("Backend not reachable")
            return 1

        r = client.post(f"{BASE}/auth/dev-login")
        r.raise_for_status()
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print(f"Logged in as {r.json()['user']['username']}")

        with PDF.open("rb") as fh:
            r = client.post(
                f"{BASE}/documents/upload",
                headers=headers,
                files={"file": (PDF.name, fh, "application/pdf")},
            )
        print("upload", r.status_code, r.text[:400])
        r.raise_for_status()
        doc_id = r.json()["id"]
        print("doc_id", doc_id)

        status = None
        for i in range(120):
            r = client.get(f"{BASE}/documents/{doc_id}/status", headers=headers)
            r.raise_for_status()
            body = r.json()
            status = body.get("status")
            print(f"ingest[{i}] status={status} chunks={body.get('chunk_count')} progress={body.get('processing_progress')}")
            if status in {"ready", "failed", "error"}:
                break
            time.sleep(2)

        if status != "ready":
            print("Ingestion failed:", status)
            return 1

        print("\n========== Q&A ==========\n")
        results = []
        for n, q in enumerate(QUESTIONS, start=1):
            print(f"Q{n}. {q}")
            try:
                r = client.post(
                    f"{BASE}/chat/ask",
                    headers=headers,
                    json={"question": q, "document_id": doc_id},
                    timeout=300.0,
                )
                if r.status_code >= 400:
                    print(f"  ERROR {r.status_code}: {r.text[:300]}\n")
                    results.append({"q": q, "ok": False, "answer": r.text[:300]})
                    continue
                data = r.json()
                answer = data.get("answer") or ""
                sources = data.get("sources") or []
                print(f"  A: {answer}")
                if sources:
                    src = sources[0]
                    print(f"  citation: {src.get('filename')} p.{src.get('page')} score={src.get('score')}")
                print()
                results.append({"q": q, "ok": True, "answer": answer, "sources": len(sources)})
            except Exception as exc:
                print(f"  EXCEPTION: {exc}\n")
                results.append({"q": q, "ok": False, "answer": str(exc)})

        ok = sum(1 for x in results if x["ok"])
        print(f"Done: {ok}/{len(results)} answered successfully")
        out = Path(__file__).resolve().parent / "sample_qa_results.json"
        out.write_text(json.dumps({"doc_id": doc_id, "pdf": str(PDF), "results": results}, indent=2), encoding="utf-8")
        print(f"Saved {out}")
        return 0 if ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
