"""
Ask UI-PV questions into the live dashboard chat session via /chat/ask
(same auth + session + document the UI uses). Saves scored results.

Does not modify documents or other sessions.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:7860/api/v1"
HERE = Path(__file__).resolve().parent
QA_PATH = HERE / "ui_verification_qa.json"
OUT_PATH = HERE / "ui_verification_results.json"
SESSION_ID = "d80d72a7-6e8a-4049-8977-bed31f3f75a3"
DOC_ID = "8c57d30c-9568-4437-b51f-b52d9ac495d1"


def login(client: httpx.Client) -> str:
    r = client.post(f"{BASE}/auth/dev-login")
    r.raise_for_status()
    return r.json()["access_token"]


def cite_pages(answer: str, sources: list) -> dict:
    pages_in_text = sorted(set(int(p) for p in re.findall(r"[Pp]age[s]?\s+(\d+)", answer)))
    source_pages = []
    for s in sources or []:
        p = s.get("page_number") or s.get("page") or s.get("page_num")
        if p is not None:
            try:
                source_pages.append(int(p))
            except Exception:
                pass
    source_pages = sorted(set(source_pages))
    return {
        "inline_citation_pages": pages_in_text,
        "source_panel_pages": source_pages,
        "has_inline_citation": bool(re.search(r"\[Source:|Page\s+\d+", answer, re.I)),
        "source_count": len(sources or []),
    }


def score(q: dict, answer: str, cite: dict) -> dict:
    low = (answer or "").lower()
    must = q.get("must_include") or []
    hits = [m for m in must if m.lower() in low]
    miss = [m for m in must if m.lower() not in low]
    expected_pages = set(q.get("source_pages") or [])
    cited = set(cite["inline_citation_pages"]) | set(cite["source_panel_pages"])
    page_overlap = sorted(expected_pages & cited) if expected_pages else []
    content_ok = len(hits) >= max(1, len(must) - 1) and len(miss) <= 1
    # soft page check: citation present is required; page overlap preferred
    cite_ok = cite["has_inline_citation"] or cite["source_count"] > 0
    passed = bool(answer) and content_ok and cite_ok
    return {
        "pass": passed,
        "must_hits": hits,
        "must_miss": miss,
        "page_overlap": page_overlap,
        "citation_ok": cite_ok,
        "content_ok": content_ok,
    }


def main() -> int:
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    questions = data["questions"]
    results = []
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        token = login(client)
        headers = {"Authorization": f"Bearer {token}"}
        for i, q in enumerate(questions, 1):
            print(f"\n=== {q['id']} ({i}/{len(questions)}) ===", flush=True)
            print(q["question"][:160], flush=True)
            if i % 4 == 1:
                token = login(client)
                headers = {"Authorization": f"Bearer {token}"}
            t0 = time.time()
            try:
                r = client.post(
                    f"{BASE}/chat/ask",
                    headers=headers,
                    json={
                        "question": q["question"],
                        "document_id": DOC_ID,
                        "session_id": SESSION_ID,
                    },
                )
                if r.status_code == 401:
                    token = login(client)
                    headers = {"Authorization": f"Bearer {token}"}
                    r = client.post(
                        f"{BASE}/chat/ask",
                        headers=headers,
                        json={
                            "question": q["question"],
                            "document_id": DOC_ID,
                            "session_id": SESSION_ID,
                        },
                    )
                elapsed = round(time.time() - t0, 1)
                if r.status_code >= 400:
                    print(f"ERROR {r.status_code}: {r.text[:300]}", flush=True)
                    results.append(
                        {
                            "id": q["id"],
                            "question": q["question"],
                            "expected": q["expected"],
                            "actual": r.text[:400],
                            "pass": False,
                            "error": r.status_code,
                            "elapsed_s": elapsed,
                        }
                    )
                    continue
                body = r.json()
                answer = body.get("answer") or ""
                sources = body.get("sources") or []
                cite = cite_pages(answer, sources)
                sc = score(q, answer, cite)
                print(f"  pass={sc['pass']} elapsed={elapsed}s sources={cite['source_count']}", flush=True)
                print(f"  A: {answer[:350].replace(chr(10), ' ')}...", flush=True)
                results.append(
                    {
                        "id": q["id"],
                        "type": q.get("type"),
                        "question": q["question"],
                        "expected": q["expected"],
                        "actual": answer,
                        "citations": cite,
                        "scoring": sc,
                        "pass": sc["pass"],
                        "elapsed_s": elapsed,
                        "source_pages_expected": q.get("source_pages"),
                    }
                )
            except Exception as exc:
                print(f"EXCEPTION: {exc}", flush=True)
                results.append(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "expected": q["expected"],
                        "actual": str(exc),
                        "pass": False,
                        "error": "exception",
                    }
                )

    passed = sum(1 for x in results if x.get("pass"))
    out = {
        "session_id": SESSION_ID,
        "document_id": DOC_ID,
        "document": "xolair_prescribing.pdf",
        "answered": f"{passed}/{len(results)}",
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_PATH} => {passed}/{len(results)} PASS", flush=True)
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
