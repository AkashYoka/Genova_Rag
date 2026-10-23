"""Complete only the Medical Avastin job (resume after upload timeout)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

# Reuse helpers / JOBS from main runner
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain_qa_run import (  # noqa: E402
    BASE,
    JOBS,
    OUT_DIR,
    login,
    wait_doc,
    wait_ready,
)

job = next(j for j in JOBS if "Avastin" in j["chat_title"])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    title = job["chat_title"]
    src: Path = job["source"]
    # Prefer local copy under sample_subsets if present
    local = Path(__file__).resolve().parents[1] / "data" / "sample_subsets" / "avastin_prescribing.pdf"
    pdf_path = local if local.exists() else src
    print(f"CHAT: {title}")
    print(f"  pdf={pdf_path} ({pdf_path.stat().st_size} bytes)", flush=True)

    # Long timeouts: upload can block while eager Celery ingests
    timeout = httpx.Timeout(connect=30.0, read=1800.0, write=1800.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        wait_ready(client)
        headers = login(client)

        r = client.post(f"{BASE}/chat/sessions", headers=headers, json={"title": title})
        r.raise_for_status()
        session_id = r.json()["id"]
        print(f"  session_id={session_id}", flush=True)

        with pdf_path.open("rb") as fh:
            r = client.post(
                f"{BASE}/documents/upload",
                headers=headers,
                files={"file": (job["upload_name"], fh, "application/pdf")},
            )
        print(f"  upload {r.status_code}: {r.text[:240]}", flush=True)
        r.raise_for_status()
        doc_id = r.json()["id"]
        print(f"  doc_id={doc_id}", flush=True)

        status_body = wait_doc(client, headers, doc_id)
        if status_body.get("status") != "ready":
            print(f"  INGEST FAILED: {status_body}", flush=True)
            return 2

        results = []
        for n, q in enumerate(job["questions"], start=1):
            print(f"\n  Q{n}. {q}", flush=True)
            headers = login(client) if n % 4 == 1 else headers
            try:
                r = client.post(
                    f"{BASE}/chat/ask",
                    headers=headers,
                    json={"question": q, "document_id": doc_id, "session_id": session_id},
                    timeout=600.0,
                )
                if r.status_code == 401:
                    headers = login(client)
                    r = client.post(
                        f"{BASE}/chat/ask",
                        headers=headers,
                        json={"question": q, "document_id": doc_id, "session_id": session_id},
                        timeout=600.0,
                    )
                if r.status_code >= 400:
                    print(f"    ERROR {r.status_code}: {r.text[:250]}", flush=True)
                    results.append({"q": q, "ok": False, "answer": r.text[:250]})
                    continue
                data = r.json()
                answer = data.get("answer") or ""
                sources = data.get("sources") or []
                print(f"    A: {answer[:500]}{'...' if len(answer) > 500 else ''}", flush=True)
                results.append({"q": q, "ok": True, "answer": answer, "sources": len(sources)})
            except Exception as exc:
                print(f"    EXCEPTION: {exc}", flush=True)
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
        print(f"\n  Saved {out_file} ({ok_n}/{len(results)} ok)", flush=True)

        # Refresh summary.json from all result files
        summary = []
        for p in sorted(OUT_DIR.glob("*.json")):
            if p.name == "summary.json":
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            answered = data.get("answered", "")
            ok_part = answered.split("/")[0] if "/" in answered else "0"
            total = answered.split("/")[1] if "/" in answered else "?"
            summary.append(
                {
                    "title": data.get("chat_title"),
                    "session_id": data.get("session_id"),
                    "document_id": data.get("document_id"),
                    "ok": ok_part == total,
                    "answered": answered,
                }
            )
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0 if ok_n == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
