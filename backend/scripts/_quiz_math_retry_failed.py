"""Re-run previously failing math quiz modes after reliability fixes."""
from __future__ import annotations

import time

import httpx

from scripts.quiz_ollama_math_qa import (
    ask,
    create_session,
    has_quiz,
    login,
    quiz_meta,
)

DOC = "e4cc3e90-5659-4bb8-bd72-8c58f41e9482"


def main() -> int:
    timeout = httpx.Timeout(connect=30.0, read=900.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        token = login(client)
        headers = {"Authorization": f"Bearer {token}"}
        sid = create_session(
            client, headers, f"Math quiz retry {time.strftime('%H:%M')}"
        )
        print("session", sid, flush=True)

        seeds = [
            "Explain the definition of the derivative from this document.",
            "From this document, what is the derivative of x^2? Show LaTeX.",
            "What is the power rule for differentiation according to this text?",
        ]
        for q in seeds:
            body = ask(
                client,
                headers,
                question=q,
                session_id=sid,
                document_id=DOC,
            )
            print(
                "seed",
                body.get("_status"),
                "quiz=",
                has_quiz(body),
                "t=",
                body.get("_elapsed_s"),
                flush=True,
            )

        body = ask(
            client,
            headers,
            question=(
                "Quiz me on our conversation with 5 questions about "
                "derivatives and equations"
            ),
            session_id=sid,
            document_id=DOC,
        )
        meta = quiz_meta(body) if has_quiz(body) else {}
        print("conversation", has_quiz(body), meta, flush=True)
        convo_ok = has_quiz(body) and meta.get("mode") == "conversation"

        body = ask(
            client,
            headers,
            question="Summarize this document and then quiz me with 5 questions",
            session_id=sid,
            document_id=DOC,
        )
        meta = quiz_meta(body) if has_quiz(body) else {}
        print("summary", has_quiz(body), meta, flush=True)
        summary_ok = has_quiz(body) and meta.get("mode") == "summary"

        print(
            "VERDICT",
            "PASS" if convo_ok and summary_ok else "FAIL",
            {"conversation": convo_ok, "summary": summary_ok},
            flush=True,
        )
        return 0 if convo_ok and summary_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
