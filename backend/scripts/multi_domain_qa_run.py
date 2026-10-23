"""
Upload math / chemistry / UPSC / medical PDFs into separate named chat sessions
and ask ~10–12 questions each via the local API (dev-login, no UI).

Large textbooks are truncated to the first N pages so local Ollama ingest finishes
in a reasonable time; the medical prescribing PDF is used in full.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
from pypdf import PdfReader, PdfWriter

BASE = "http://127.0.0.1:7860/api/v1"
OUT_DIR = Path(__file__).resolve().parent / "multi_domain_results"
SUBSET_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_subsets"
PAGE_LIMIT = 25  # first N pages for huge textbooks


def subset_pdf(src: Path, dest: Path, max_pages: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(src))
    writer = PdfWriter()
    n = min(len(reader.pages), max_pages)
    for i in range(n):
        writer.add_page(reader.pages[i])
    with dest.open("wb") as fh:
        writer.write(fh)
    print(f"  subset {src.name}: {n}/{len(reader.pages)} pages -> {dest.name}")
    return dest


# Prefer pre-built excerpts under data/sample_subsets (first ~20 pages).
JOBS = [
    {
        "chat_title": "Math — Calculus Volume 1",
        "source": Path(r"C:\PC\Yoka\Misc\genova-rag\samples\Education\calculus-volume-1_-_WEB.pdf"),
        "prebuilt": SUBSET_DIR / "math-calculus-text.pdf",
        "subset": True,
        "upload_name": "math-calculus-text.pdf",
        "questions": [
            "What is this document about?",
            "What topics are covered in the table of contents or early chapters?",
            "Explain the main idea introduced in the opening chapter.",
            "What is a function according to this text?",
            "Define the limit of a function as described in the document.",
            "What does continuity of a function mean here?",
            "Give an example problem or illustration from the early pages.",
            "What notation does the book use for derivatives, if introduced?",
            "Summarize any learning objectives stated at the start.",
            "List key mathematical terms introduced in the first few pages.",
            "Who is the intended audience of this textbook?",
            "What prerequisites does the book assume?",
        ],
    },
    {
        "chat_title": "Chemistry — Learning Resources",
        "source": Path(r"C:\PC\Yoka\Misc\genova-rag\samples\Education\Chemistry-LR.pdf"),
        "prebuilt": SUBSET_DIR / "chemistry-text.pdf",
        "subset": True,
        "upload_name": "chemistry-text.pdf",
        "questions": [
            "What is this chemistry document?",
            "What units or chapters appear early in the book?",
            "What is an atom according to this text?",
            "Explain the periodic table as described here.",
            "What are elements and compounds?",
            "Describe chemical bonding if covered in these pages.",
            "What is a mole or molar mass if mentioned?",
            "List any important definitions from the opening sections.",
            "What experimental or lab safety notes appear, if any?",
            "Summarize the first major chemistry concept taught.",
            "Give an example reaction or formula from the document.",
            "Who would benefit from reading this material?",
        ],
    },
    {
        "chat_title": "UPSC Prelims Complete Guide",
        "source": Path(r"C:\PC\Yoka\Misc\genova-rag\samples\Education\upsc-prelims-complete.pdf"),
        "prebuilt": SUBSET_DIR / "upsc-text.pdf",
        "subset": True,
        "upload_name": "upsc-text.pdf",
        "questions": [
            "What is this document and who is it for?",
            "What is the UPSC Civil Services Preliminary exam structure according to this PDF?",
            "How many papers are there in the Prelims and what do they cover?",
            "What is the marking scheme or negative marking mentioned?",
            "List the major syllabus areas for General Studies Paper I.",
            "What does CSAT / Paper II cover?",
            "What eligibility criteria are mentioned?",
            "What exam strategy tips are given in the early pages?",
            "Are there recommended books or resources listed?",
            "Summarize the cut-off or selection process if described.",
            "What is the difference between Prelims and Mains here?",
            "List any important dates or stages of the selection process mentioned.",
        ],
    },
    {
        "chat_title": "Medical — Avastin Prescribing Info",
        "source": Path(r"C:\PC\Yoka\Misc\genova-rag\samples\Medcare\avastin_prescribing.pdf"),
        "subset": False,
        "upload_name": "avastin_prescribing.pdf",
        "questions": [
            "What drug is this prescribing information for?",
            "What is the generic / active ingredient name?",
            "What are the approved indications for Avastin?",
            "What is the recommended dosage for one major indication?",
            "List important warnings or boxed warnings.",
            "What are common adverse reactions?",
            "Who should not take this medication (contraindications)?",
            "How is Avastin administered?",
            "Are there special monitoring requirements?",
            "What drug interactions are noted?",
            "Summarize pregnancy or lactation guidance if present.",
            "What is the mechanism of action described?",
        ],
    },
]


def wait_ready(client: httpx.Client) -> None:
    for _ in range(60):
        try:
            if client.get("http://127.0.0.1:7860/api/health").status_code == 200:
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
            f"progress={body.get('processing_progress')} stage={body.get('processing_stage')}"
        )
        if status in {"ready", "failed", "error"}:
            return body
        time.sleep(3)
    raise TimeoutError(f"Ingest timed out for {doc_id}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    with httpx.Client(timeout=600.0) as client:
        wait_ready(client)
        headers = login(client)
        print()

        for job in JOBS:
            title = job["chat_title"]
            print("=" * 72)
            print(f"CHAT: {title}")
            print("=" * 72)

            # Refresh token each job — long LLM runs can outlive JWT expiry
            headers = login(client)

            src: Path = job["source"]
            if not src.exists():
                print(f"  MISSING FILE: {src}")
                summary.append({"title": title, "ok": False, "error": "missing file"})
                continue

            prebuilt = job.get("prebuilt")
            if prebuilt and Path(prebuilt).exists():
                pdf_path = Path(prebuilt)
                print(f"  using prebuilt excerpt: {pdf_path.name} ({pdf_path.stat().st_size} bytes)", flush=True)
            elif job["subset"]:
                pdf_path = subset_pdf(src, SUBSET_DIR / job["upload_name"], PAGE_LIMIT)
            else:
                pdf_path = src
                print(f"  using full PDF: {src.name}", flush=True)

            # Named chat session
            r = client.post(f"{BASE}/chat/sessions", headers=headers, json={"title": title})
            r.raise_for_status()
            session_id = r.json()["id"]
            print(f"  session_id={session_id}")

            with pdf_path.open("rb") as fh:
                r = client.post(
                    f"{BASE}/documents/upload",
                    headers=headers,
                    files={"file": (job["upload_name"], fh, "application/pdf")},
                )
            print(f"  upload {r.status_code}: {r.text[:240]}")
            if r.status_code >= 400:
                summary.append({"title": title, "ok": False, "error": r.text[:400]})
                continue
            doc_id = r.json()["id"]
            print(f"  doc_id={doc_id}")

            status_body = wait_doc(client, headers, doc_id)
            if status_body.get("status") != "ready":
                print(f"  INGEST FAILED: {status_body}")
                summary.append({"title": title, "ok": False, "error": status_body})
                continue

            results = []
            for n, q in enumerate(job["questions"], start=1):
                print(f"\n  Q{n}. {q}")
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
                        print("    token expired — re-login and retry", flush=True)
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
                        print(f"    ERROR {r.status_code}: {r.text[:250]}")
                        results.append({"q": q, "ok": False, "answer": r.text[:250]})
                        continue
                    data = r.json()
                    answer = data.get("answer") or ""
                    sources = data.get("sources") or []
                    print(f"    A: {answer[:500]}{'...' if len(answer) > 500 else ''}")
                    if sources:
                        s0 = sources[0]
                        print(
                            f"    citation: {s0.get('filename')} p.{s0.get('page')} "
                            f"score={s0.get('score')}"
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
                    print(f"    EXCEPTION: {exc}")
                    results.append({"q": q, "ok": False, "answer": str(exc)})

            ok_n = sum(1 for x in results if x["ok"])
            job_out = {
                "chat_title": title,
                "session_id": session_id,
                "document_id": doc_id,
                "pdf": str(pdf_path),
                "source_pdf": str(src),
                "answered": f"{ok_n}/{len(results)}",
                "results": results,
            }
            out_file = OUT_DIR / (title.replace(" ", "_").replace("—", "-") + ".json")
            out_file.write_text(json.dumps(job_out, indent=2), encoding="utf-8")
            print(f"\n  Saved {out_file} ({ok_n}/{len(results)} ok)\n")
            summary.append(
                {
                    "title": title,
                    "session_id": session_id,
                    "document_id": doc_id,
                    "ok": ok_n == len(results),
                    "answered": f"{ok_n}/{len(results)}",
                }
            )

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 72)
    print("SUMMARY")
    for row in summary:
        print(f"  - {row.get('title')}: {row.get('answered') or row.get('error')}")
    print(f"Wrote {summary_path}")
    return 0 if all(r.get("ok") for r in summary) else 2


if __name__ == "__main__":
    sys.exit(main())
