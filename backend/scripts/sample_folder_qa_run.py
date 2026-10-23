"""Upload ALL PDFs under project sample/, ask 12-15 questions each, save results JSON."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fitz
import httpx
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

BASE = "http://127.0.0.1:7860/api/v1"
HEALTH = "http://127.0.0.1:7860/api/health"
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample"
SUBSETS = Path(__file__).resolve().parents[1] / "data" / "sample_subsets"
OUT_FILE = Path(__file__).resolve().parent / "sample_folder_qa_results.json"
MAX_UPLOAD = 5 * 1024 * 1024
EXCERPT_PAGES = 25

UPSC_SESSION = "UPSC Prelims — sample folder"

UPSC_QS = [
    "What is this document about?",
    "What exam or syllabus does this material cover?",
    "What subjects or papers are mentioned in the early pages?",
    "List the major topics or sections introduced.",
    "What is the structure of the UPSC Prelims exam as described?",
    "Are there any important instructions or guidelines for candidates?",
    "What historical or polity topics appear early in the text?",
    "Mention any geography or environment topics covered.",
    "What economy or current-affairs related points appear?",
    "Are practice questions or previous-year style items present?",
    "What dates, editions, or publication details are stated?",
    "Who is the intended audience for this material?",
    "Summarize the first major concept or chapter theme.",
    "List key terms a Prelims aspirant should note from these pages.",
    "What should a reader remember after skimming this excerpt?",
]

CALC_QS = [
    "What is this document about?",
    "What mathematical subject or volume is this?",
    "What topics appear in the opening chapters?",
    "Explain the main idea introduced early on.",
    "List key terms or definitions from the first pages.",
    "Give one concrete example or illustration from the excerpt.",
    "What learning objectives or goals are stated?",
    "Who is the intended audience?",
    "What prerequisites does the material assume?",
    "Summarize the first major concept taught.",
    "What notation or formulas are introduced?",
    "Are practice exercises mentioned?",
    "What should a student take away from these pages?",
    "List any theorems or properties introduced early.",
]

CHEM_QS = [
    "What is this document about?",
    "What chemistry topics appear in the early pages?",
    "Explain the main idea introduced in the opening section.",
    "List key chemical terms or definitions.",
    "Give one concrete example or data point from the excerpt.",
    "What learning objectives are stated if any?",
    "Who is the intended audience?",
    "What formulas or equations are introduced?",
    "Summarize the first major concept taught.",
    "Are practice questions or exercises mentioned?",
    "What units or standards are referenced?",
    "List important warnings or notes if present.",
    "What should a student remember from these pages?",
    "What chapters or sections are outlined early?",
]

XOLAIR_QS = [
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
    "What age groups is this approved for?",
    "What should clinicians remember before prescribing?",
]


def questions_for(name: str) -> list[str]:
    n = name.lower()
    if "upsc" in n:
        return UPSC_QS
    if "calculus" in n or "math" in n:
        return CALC_QS
    if "chem" in n:
        return CHEM_QS
    if "xolair" in n or "prescrib" in n:
        return XOLAIR_QS
    return UPSC_QS[:12]


def session_title_for(name: str) -> str:
    if "upsc" in name.lower():
        return UPSC_SESSION
    return f"Sample folder — {name}"


def extract_text(src: Path, max_pages: int) -> str:
    doc = fitz.open(src)
    n = min(len(doc), max_pages)
    parts: list[str] = []
    for i in range(n):
        t = (doc[i].get_text("text") or "").replace("\x00", "").strip()
        if t:
            parts.append(f"--- Page {i + 1} ---\n{t}")
    doc.close()
    return "\n\n".join(parts)


def write_text_pdf(text: str, dest: Path) -> None:
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize = 9
    body.leading = 11
    story = []
    for block in text.split("\n"):
        safe = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if not safe.strip():
            story.append(Spacer(1, 6))
            continue
        story.append(Paragraph(safe, body))
    SimpleDocTemplate(
        str(dest),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    ).build(story)


def reuse_or_build_excerpt(src: Path) -> Path:
    """Prefer existing slim text excerpt; else build first N pages as text PDF."""
    SUBSETS.mkdir(parents=True, exist_ok=True)
    name = src.name.lower()
    known = {
        "upsc-prelims-complete.pdf": [
            Path(__file__).resolve().parents[2]
            / "samples"
            / "Education"
            / "upsc-prelims_excerpt.pdf",
            SUBSETS / "upsc-text.pdf",
        ],
        "calculus-volume-1_-_web-1-300.pdf": [SUBSETS / "math-calculus-text.pdf"],
        "chemistry-lr-1-355.pdf": [SUBSETS / "chemistry-text.pdf"],
    }
    for cand in known.get(name, []):
        if cand.exists() and cand.stat().st_size < MAX_UPLOAD:
            print(f"  reusing excerpt {cand} ({cand.stat().st_size} bytes)", flush=True)
            return cand

    dest = SUBSETS / f"{src.stem}_text_excerpt.pdf"
    if dest.exists() and dest.stat().st_size < MAX_UPLOAD:
        print(f"  reusing {dest}", flush=True)
        return dest

    print(f"  building text excerpt first {EXCERPT_PAGES} pages -> {dest.name}", flush=True)
    text = extract_text(src, EXCERPT_PAGES)
    write_text_pdf(text, dest)
    print(f"  wrote {dest} ({dest.stat().st_size} bytes, chars={len(text)})", flush=True)
    return dest


def upload_path_for(src: Path) -> Path:
    if src.stat().st_size <= MAX_UPLOAD:
        return src
    return reuse_or_build_excerpt(src)


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
    for i in range(400):
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
    pdfs = sorted(p for p in SAMPLE_DIR.glob("*.pdf") if p.is_file())
    print(f"Found {len(pdfs)} PDFs under {SAMPLE_DIR}", flush=True)
    for p in pdfs:
        print(f"  - {p.name} ({p.stat().st_size:,} bytes)", flush=True)

    all_jobs: list[dict] = []

    with httpx.Client(timeout=600.0) as client:
        wait_ready(client)
        headers = login(client)

        for pdf in pdfs:
            title = session_title_for(pdf.name)
            print("\n" + "=" * 72, flush=True)
            print(f"PDF: {pdf.name} -> session '{title}'", flush=True)
            print("=" * 72, flush=True)

            try:
                upload_pdf = upload_path_for(pdf)
                headers = login(client)

                r = client.post(
                    f"{BASE}/chat/sessions",
                    headers=headers,
                    json={"title": title},
                )
                r.raise_for_status()
                session_id = r.json()["id"]
                print(f"  session_id={session_id}", flush=True)

                with upload_pdf.open("rb") as fh:
                    r = client.post(
                        f"{BASE}/documents/upload",
                        headers=headers,
                        files={"file": (upload_pdf.name, fh, "application/pdf")},
                    )
                print(f"  upload {r.status_code}: {r.text[:240]}", flush=True)
                if r.status_code >= 400:
                    all_jobs.append(
                        {
                            "source_pdf": str(pdf),
                            "uploaded_pdf": str(upload_pdf),
                            "chat_title": title,
                            "ok": False,
                            "error": r.text[:400],
                            "answered": "0/0",
                            "results": [],
                        }
                    )
                    continue

                doc_id = r.json()["id"]
                print(f"  doc_id={doc_id}", flush=True)
                status_body = wait_doc(client, headers, doc_id)
                if status_body.get("status") != "ready":
                    all_jobs.append(
                        {
                            "source_pdf": str(pdf),
                            "uploaded_pdf": str(upload_pdf),
                            "chat_title": title,
                            "session_id": session_id,
                            "document_id": doc_id,
                            "ok": False,
                            "error": status_body,
                            "answered": "0/0",
                            "results": [],
                        }
                    )
                    continue

                results = []
                for n, q in enumerate(questions_for(pdf.name), start=1):
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
                        preview = f"    A: {answer[:500]}{'...' if len(answer) > 500 else ''}"
                        try:
                            print(preview, flush=True)
                        except UnicodeEncodeError:
                            print(preview.encode("ascii", "replace").decode("ascii"), flush=True)
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
                job = {
                    "source_pdf": str(pdf),
                    "uploaded_pdf": str(upload_pdf),
                    "chat_title": title,
                    "session_id": session_id,
                    "document_id": doc_id,
                    "ok": ok_n == len(results) and len(results) > 0,
                    "answered": f"{ok_n}/{len(results)}",
                    "results": results,
                }
                all_jobs.append(job)
                print(f"\n  Done {ok_n}/{len(results)}", flush=True)
            except Exception as exc:
                print(f"  FATAL: {exc}", flush=True)
                all_jobs.append(
                    {
                        "source_pdf": str(pdf),
                        "chat_title": title,
                        "ok": False,
                        "error": str(exc),
                        "answered": "0/0",
                        "results": [],
                    }
                )

    payload = {
        "sample_dir": str(SAMPLE_DIR),
        "jobs": all_jobs,
        "totals": {
            "pdfs": len(pdfs),
            "jobs_ok": sum(1 for j in all_jobs if j.get("ok")),
            "questions_ok": sum(
                sum(1 for r in j.get("results", []) if r.get("ok")) for j in all_jobs
            ),
            "questions_total": sum(len(j.get("results", [])) for j in all_jobs),
        },
    }
    OUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}", flush=True)
    for j in all_jobs:
        print(
            f"  - {j.get('chat_title')}: {j.get('answered')} "
            f"{'(ok)' if j.get('ok') else '(FAIL)'}",
            flush=True,
        )
    return 0 if all_jobs and all(j.get("ok") for j in all_jobs) else 2


if __name__ == "__main__":
    sys.exit(main())
