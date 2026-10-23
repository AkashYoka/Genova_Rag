"""One-shot local verification: register, upload PDF, ask via Ollama RAG."""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import httpx
from reportlab.pdfgen import canvas

BASE = "http://127.0.0.1:7860/api/v1"
ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "test_sample.pdf"


def main() -> int:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(PDF_PATH))
    c.drawString(72, 720, "PDF Assistant RAG local Ollama verification document.")
    c.drawString(72, 700, "The capital of the fictional nation of Averia is Silverport.")
    c.drawString(72, 680, "Averia exports rare earth magnets and citrus fruit.")
    c.save()
    print(f"wrote {PDF_PATH}")

    suffix = uuid.uuid4().hex[:8]
    email = f"ollama_test_{suffix}@example.com"
    username = f"ollamauser{suffix}"
    password = "TestPass123!"

    with httpx.Client(timeout=300.0) as client:
        r = client.post(
            f"{BASE}/auth/register",
            json={"email": email, "username": username, "password": password},
        )
        print("register", r.status_code, r.text[:500])
        r.raise_for_status()

        r = client.post(
            f"{BASE}/auth/login",
            json={"email": email, "password": password},
        )
        print("login", r.status_code, r.text[:300])
        r.raise_for_status()
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        with PDF_PATH.open("rb") as fh:
            r = client.post(
                f"{BASE}/documents/upload",
                headers=headers,
                files={"file": (PDF_PATH.name, fh, "application/pdf")},
            )
        print("upload", r.status_code, r.text[:800])
        r.raise_for_status()
        payload = r.json()
        doc_id = payload.get("id") or payload.get("document_id")
        print("doc_id", doc_id)

        status = None
        for i in range(90):
            r = client.get(f"{BASE}/documents/{doc_id}/status", headers=headers)
            r.raise_for_status()
            status = r.json().get("status")
            print(f"status[{i}]", status, r.text[:200])
            if status in {"ready", "failed", "error"}:
                break
            time.sleep(2)

        if status != "ready":
            print("INGEST_FAILED", status)
            return 1

        # Confirm chroma path exists for ollama nomic
        chroma = ROOT / "data" / "chroma_db_ollama_nomic"
        print("chroma_dir_exists", chroma.exists(), chroma)

        r = client.post(
            f"{BASE}/chat/ask",
            headers=headers,
            json={"question": "What is the capital of Averia?", "document_id": doc_id},
        )
        print("ask", r.status_code, r.text[:2000])
        if r.status_code >= 400:
            # Try stream endpoint as fallback signal
            r2 = client.post(
                f"{BASE}/chat/ask/stream",
                headers=headers,
                json={"question": "What is the capital of Averia?", "document_id": doc_id},
            )
            print("ask_stream", r2.status_code, r2.text[:2000])
            return 1

        data = r.json()
        answer = (data.get("answer") or "").lower()
        sources = data.get("sources") or []
        print("answer", data.get("answer"))
        print("sources_count", len(sources))
        ok = "silverport" in answer
        print("VERDICT", "PASS" if ok else "CHECK_MANUALLY")
        return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
