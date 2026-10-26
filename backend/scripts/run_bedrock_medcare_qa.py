"""Run medical-care Bedrock Nova 2 Lite RAG questions and persist them to a chat session.

Uses AWS_BEARER_TOKEN_BEDROCK from the environment / backend/.env.
Does not print secrets.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv

load_dotenv(BACKEND_ROOT / ".env")

from app.database import SessionLocal
from app.models import Document, User, ChatSession, ChatMessage
from app.rag.agent import generate_answer
from app.rag.bedrock_client import (
    DEFAULT_NOVA_2_LITE_PROFILE,
    NOVA_2_LITE_INPUT_USD_PER_1M,
    NOVA_2_LITE_OUTPUT_USD_PER_1M,
)
from app.routes.chat import _save_message

QUESTIONS = [
    "What are the approved indications for XOLAIR?",
    "What is the recommended dosing for XOLAIR in chronic spontaneous urticaria?",
    "What serious allergic reactions or anaphylaxis warnings are listed for XOLAIR?",
    "Who should not receive XOLAIR (contraindications)?",
    "What laboratory monitoring or IgE testing is recommended before starting XOLAIR?",
    "What are common adverse reactions reported with XOLAIR?",
    "Can XOLAIR be self-administered, and under what conditions?",
    "What pregnancy or lactation considerations are mentioned for XOLAIR?",
]


def pick_xolair_doc(db, user_id: str):
    docs = (
        db.query(Document)
        .filter(
            Document.user_id == user_id,
            Document.is_deleted.is_(False),
            Document.status == "ready",
        )
        .all()
    )
    for doc in docs:
        name = (doc.original_name or "").lower()
        if "xolair" in name:
            return doc
    for doc in docs:
        name = (doc.original_name or "").lower()
        if "prescrib" in name:
            return doc
    return docs[0] if docs else None


def _assistant_payload(answer: str, sources: list, usage: dict | None) -> str:
    """Persist answer; embed usage metadata in a trailing marker for UI reload.

    The dashboard history API only returns content/sources today, so we keep the
    visible answer clean and append a machine-readable HTML comment the frontend
    can ignore (sources still carry citation chips).
    """
    return answer


def main() -> int:
    db = SessionLocal()
    try:
        user = (
            db.query(User).filter(User.email == "localdev@localhost").first()
            or db.query(User).order_by(User.created_at.desc()).first()
        )
        if not user:
            print("ERROR: no user in database")
            return 1
        doc = pick_xolair_doc(db, user.id)
        if not doc:
            print("ERROR: no ready prescribing document found")
            return 1

        session = ChatSession(
            user_id=user.id,
            title="Bedrock Nova 2 Lite — Medcare",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        print("model_id:", DEFAULT_NOVA_2_LITE_PROFILE)
        print(
            "pricing: ${}/1M in + ${}/1M out (estimated; not billed)".format(
                NOVA_2_LITE_INPUT_USD_PER_1M, NOVA_2_LITE_OUTPUT_USD_PER_1M
            )
        )
        print("document:", doc.original_name, doc.id)
        print("session:", session.id, session.title)
        print("---")

        results = []
        cumulative = 0.0
        for i, question in enumerate(QUESTIONS, start=1):
            started = time.perf_counter()
            try:
                result = generate_answer(
                    question=question,
                    user_id=user.id,
                    document_id=doc.id,
                    llm_provider="bedrock",
                    chat_history=[],
                )
            except Exception as exc:
                print(f"Q{i} FAILED: {type(exc).__name__}: {exc}")
                results.append({"question": question, "error": str(exc)})
                continue

            elapsed = round((time.perf_counter() - started) * 1000)
            usage = result.get("usage") or {}
            cost = usage.get("estimated_cost_usd")
            if isinstance(cost, (int, float)):
                cumulative += float(cost)

            answer = (result.get("answer") or "").strip()
            sources = result.get("sources") or []

            # Persist both turns so the dashboard session shows the full thread.
            _save_message(
                db,
                user.id,
                doc.id,
                "user",
                question,
                session_id=session.id,
            )
            # Embed usage JSON into sources_json alongside sources for history reload.
            enriched_sources = {
                "chunks": sources,
                "usage": usage or None,
                "llm_provider": "bedrock",
                "model_id": usage.get("model_id", DEFAULT_NOVA_2_LITE_PROFILE),
                "response_time_ms": elapsed,
            }
            msg = ChatMessage(
                user_id=user.id,
                document_id=doc.id,
                session_id=session.id,
                role="assistant",
                content=_assistant_payload(answer, sources, usage),
                sources_json=json.dumps(enriched_sources),
            )
            db.add(msg)
            db.commit()

            row = {
                "n": i,
                "question": question,
                "answer_preview": answer[:240].replace("\n", " "),
                "sources": len(sources),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "estimated_cost_usd": cost,
                "cost_type": usage.get("cost_type", "estimated"),
                "model_id": usage.get("model_id", DEFAULT_NOVA_2_LITE_PROFILE),
                "elapsed_ms": elapsed,
                "cumulative_estimated_usd": cumulative,
            }
            results.append(row)
            print(
                f"Q{i}: in={row['input_tokens']} out={row['output_tokens']} "
                f"tot={row['total_tokens']} est=${cost} "
                f"cum_est=${cumulative:.6f} ({elapsed}ms) sources={row['sources']}"
            )
            print(f"   {row['answer_preview']}...")

        msg_count = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session.id)
            .count()
        )
        print("---")
        print("persisted_messages:", msg_count, "(expected", len(QUESTIONS) * 2, ")")
        print("session_id:", session.id)

        out = BACKEND_ROOT / "scripts" / "bedrock_medcare_qa_results.json"
        out.write_text(
            json.dumps(
                {
                    "model_id": DEFAULT_NOVA_2_LITE_PROFILE,
                    "pricing_note": (
                        f"Estimated using ${NOVA_2_LITE_INPUT_USD_PER_1M}/1M input + "
                        f"${NOVA_2_LITE_OUTPUT_USD_PER_1M}/1M output; not an AWS bill."
                    ),
                    "document": {"id": doc.id, "name": doc.original_name},
                    "session_id": session.id,
                    "persisted_messages": msg_count,
                    "cumulative_estimated_usd": cumulative,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("wrote", out)
        print("cumulative_estimated_usd", f"{cumulative:.6f}")
        ok = (
            results
            and all("error" not in r for r in results)
            and msg_count == len(QUESTIONS) * 2
        )
        return 0 if ok else 2
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
