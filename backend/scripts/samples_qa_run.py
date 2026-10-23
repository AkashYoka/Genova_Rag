"""
Upload ALL PDFs under project samples/ and ask 10-12 questions each via local API.
Uses /api/v1/auth/dev-login — no .env, no UI.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:7860/api/v1"
HEALTH = "http://127.0.0.1:7860/api/health"
SAMPLES = Path(__file__).resolve().parents[2] / "samples"
OUT_DIR = Path(__file__).resolve().parent / "samples_qa_results"
MAX_BYTES = 45 * 1024 * 1024

GENERIC_QS = [
    "What is this document about?",
    "What is the title or name of this document?",
    "Who is the author, publisher, or issuing organization if mentioned?",
    "Summarize the main topics covered.",
    "List key terms or definitions introduced.",
    "What audience is this document intended for?",
    "What important warnings, notes, or caveats appear?",
    "Give one concrete example or data point from the document.",
    "What dates, versions, or revision information are present?",
    "What conclusions or recommendations does the document make?",
    "List any tables, figures, or numbered sections referenced early on.",
    "What should a reader remember after skimming this document?",
]

MEDCARE_QS = [
    "What drug is this prescribing information for?",
    "What is the generic or active ingredient name?",
    "What are the approved indications?",
    "What is the recommended dosage for one major indication?",
    "List important warnings or boxed warnings.",
    "What are common adverse reactions?",
    "What are the contraindications?",
    "How is this medication administered?",
    "Are there special monitoring requirements?",
    "What drug interactions are noted?",
    "Summarize pregnancy or lactation guidance if present.",
    "What is the mechanism of action described?",
]

PAYMENT_QS = [
    "What is this document?",
    "Who is the vendor or supplier?",
    "Who is the buyer or bill-to party?",
    "What is the invoice or document number?",
    "What is the invoice or document date?",
    "What goods or services are listed?",
    "What is the taxable or subtotal amount before tax?",
    "What tax amounts (CGST/SGST/IGST) are charged?",
    "What is the total amount including tax?",
    "What currency is used?",
    "Is a purchase order (PO) number mentioned?",
    "What GST rates apply if stated?",
]

EDU_QS = [
    "What is this document about?",
    "What topics appear in the table of contents or early chapters?",
    "Explain the main idea introduced in the opening section.",
    "List key terms defined in the early pages.",
    "Give an example problem or illustration from the excerpt.",
    "What learning objectives or goals are stated?",
    "Who is the intended audience?",
    "What prerequisites does the material assume?",
    "Summarize the first major concept taught.",
    "What notation or formulas are introduced?",
    "Are there practice questions or exercises mentioned?",
    "What should a student take away from these pages?",
]


def questions_for(path: Path) -> list[str]:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "medcare" in parts or "prescrib" in name:
        return MEDCARE_QS
    if "payment" in parts or "invoice" in name or "isi" in name:
        return PAYMENT_QS
    if "education" in parts or "excerpt" in name:
        return EDU_QS
    return GENERIC_QS


def safe_name(title: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", title).strip("_")
    return s[:120] or "doc"


def wait_ready(client: httpx.Client) -> None:
    for _ in range(60):
        try:
            if client.get(HEALTH).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Backend not reachable")


def login(client: httpx.Client) -> dict:
    r = client.post(f"{BASE}/auth/dev-login")
    r.raise_for_status()
    token = r.json()["access_token"]
    print(f"Logged in as {r.json()['user']['username']}", flush=True)
    return {"Authorization": f"Bearer {token}"}


def wait_doc(client: httpx.Client, headers: dict, doc_id: str) -> dict:
    for i in range(600):
        r = client.get(f"{BASE}/documents/{doc_id}/status", headers=headers)
        r.raise_for_status()
        body = r.json()
        status = body.get("status")
        print(
            f"    ingest[{i}] status={status} chunks={body.get('chunk_count')} "
            f"progress={body.get('processing_progress')} stage={body.get('processing_stage')}",
            flush=True,
        )
        if status in {"ready", "failed", "error"}:
            return body
        time.sleep(3)
    raise TimeoutError(f"Ingest timed out for {doc_id}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SAMPLES.exists():
        print(f"Samples folder missing: {SAMPLES}", flush=True)
        return 1

    pdfs = sorted(p for p in SAMPLES.rglob("*.pdf") if p.is_file())
    print(f"Found {len(pdfs)} PDFs under {SAMPLES}", flush=True)
    summary: list[dict] = []

    with httpx.Client(timeout=600.0) as client:
        wait_ready(client)
        headers = login(client)

        for pdf in pdfs:
            title = pdf.name
            rel = str(pdf.relative_to(SAMPLES))
            print("\n" + "=" * 72, flush=True)
            print(f"PDF: {rel}", flush=True)
            print("=" * 72, flush=True)

            size = pdf.stat().st_size
            if size > MAX_BYTES:
                print(f"  SKIP >45MB ({size:,} bytes)", flush=True)
                summary.append(
                    {
                        "pdf": rel,
                        "title": title,
                        "ok": False,
                        "error": f"skipped size {size}",
                        "answered": "0/0",
                    }
                )
                continue

            headers = login(client)

            try:
                r = client.post(
                    f"{BASE}/chat/sessions",
                    headers=headers,
                    json={"title": title},
                )
                r.raise_for_status()
                session_id = r.json()["id"]
                print(f"  session_id={session_id}", flush=True)

                with pdf.open("rb") as fh:
                    r = client.post(
                        f"{BASE}/documents/upload",
                        headers=headers,
                        files={"file": (pdf.name, fh, "application/pdf")},
                    )
                print(f"  upload {r.status_code}: {r.text[:240]}", flush=True)
                if r.status_code >= 400:
                    summary.append(
                        {
                            "pdf": rel,
                            "title": title,
                            "ok": False,
                            "error": r.text[:400],
                            "answered": "0/0",
                        }
                    )
                    continue
                doc_id = r.json()["id"]
                print(f"  doc_id={doc_id}", flush=True)

                status_body = wait_doc(client, headers, doc_id)
                if status_body.get("status") != "ready":
                    print(f"  INGEST FAILED: {status_body}", flush=True)
                    summary.append(
                        {
                            "pdf": rel,
                            "title": title,
                            "session_id": session_id,
                            "document_id": doc_id,
                            "ok": False,
                            "error": status_body,
                            "answered": "0/0",
                        }
                    )
                    continue

                results = []
                for n, q in enumerate(questions_for(pdf), start=1):
                    print(f"\n  Q{n}. {q}", flush=True)
                    try:
                        r = client.post(
                            f"{BASE}/chat/ask",
                            headers=headers,
                            json={
                                "question": q,
                                "document_id": doc_id,
                                "session_id": session_id,
                            },
                            timeout=600.0,
                        )
                        if r.status_code == 401:
                            print("    token expired — re-login", flush=True)
                            headers = login(client)
                            r = client.post(
                                f"{BASE}/chat/ask",
                                headers=headers,
                                json={
                                    "question": q,
                                    "document_id": doc_id,
                                    "session_id": session_id,
                                },
                                timeout=600.0,
                            )
                        if r.status_code >= 400:
                            print(f"    ERROR {r.status_code}: {r.text[:250]}", flush=True)
                            results.append({"q": q, "ok": False, "answer": r.text[:250]})
                            continue
                        data = r.json()
                        answer = data.get("answer") or ""
                        sources = data.get("sources") or []
                        print(
                            f"    A: {answer[:500]}{'...' if len(answer) > 500 else ''}",
                            flush=True,
                        )
                        results.append(
                            {
                                "q": q,
                                "ok": True,
                                "answer": answer,
                                "sources": len(sources),
                            }
                        )
                    except Exception as exc:
                        print(f"    EXCEPTION: {exc}", flush=True)
                        results.append({"q": q, "ok": False, "answer": str(exc)})

                ok_n = sum(1 for x in results if x["ok"])
                job_out = {
                    "chat_title": title,
                    "session_id": session_id,
                    "document_id": doc_id,
                    "pdf": str(pdf),
                    "answered": f"{ok_n}/{len(results)}",
                    "results": results,
                }
                out_file = OUT_DIR / f"{safe_name(title)}.json"
                out_file.write_text(json.dumps(job_out, indent=2), encoding="utf-8")
                print(f"\n  Saved {out_file} ({ok_n}/{len(results)} ok)", flush=True)
                summary.append(
                    {
                        "pdf": rel,
                        "title": title,
                        "session_id": session_id,
                        "document_id": doc_id,
                        "ok": ok_n == len(results),
                        "answered": f"{ok_n}/{len(results)}",
                        "result_file": str(out_file),
                    }
                )
            except Exception as exc:
                print(f"  FATAL: {exc}", flush=True)
                summary.append(
                    {
                        "pdf": rel,
                        "title": title,
                        "ok": False,
                        "error": str(exc),
                        "answered": "0/0",
                    }
                )

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 72, flush=True)
    print("SUMMARY", flush=True)
    for row in summary:
        print(
            f"  - {row.get('title')}: {row.get('answered')} "
            f"{'(ok)' if row.get('ok') else '(FAIL)'} "
            f"{row.get('error', '')}",
            flush=True,
        )
    print(f"Wrote {summary_path}", flush=True)
    return 0 if summary and all(r.get("ok") for r in summary) else 2


if __name__ == "__main__":
    sys.exit(main())
