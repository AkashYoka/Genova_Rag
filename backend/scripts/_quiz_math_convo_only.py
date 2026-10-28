"""Isolate conversation-quiz generation for math PDF."""
from __future__ import annotations

import time

import httpx

from scripts.quiz_ollama_math_qa import ask, create_session, has_quiz, login, quiz_meta

DOC = "e4cc3e90-5659-4bb8-bd72-8c58f41e9482"


def main() -> int:
    timeout = httpx.Timeout(connect=30.0, read=900.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        headers = {"Authorization": f"Bearer {login(client)}"}
        sid = create_session(client, headers, f"Math convo only {time.strftime('%H:%M')}")
        for q in [
            "What is the power rule?",
            "What is the derivative of x^2 using the power rule?",
        ]:
            b = ask(client, headers, question=q, session_id=sid, document_id=DOC)
            print("seed", (b.get("answer") or "")[:120].replace("\n", " "), flush=True)

        b = ask(
            client,
            headers,
            question="Quiz me on our conversation with 5 questions about derivatives",
            session_id=sid,
            document_id=DOC,
        )
        print("status", b.get("_status"), flush=True)
        print("quiz", has_quiz(b), quiz_meta(b) if has_quiz(b) else None, flush=True)
        print("answer", (b.get("answer") or b.get("raw") or "")[:500], flush=True)
        return 0 if has_quiz(b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
