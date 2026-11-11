"""
Chat routes — ask questions with RAG, stream responses via SSE, manage history.
"""

import asyncio
import html
import json
import time
from datetime import datetime, timezone
from io import BytesIO
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.cache import get_cached_payload, get_cached_response, set_cached_response
from app.database import get_db
from app.exceptions import (
    AppException,
    NotFoundException,
    UnauthorizedException,
    ValidationException,
)
from app.metrics import record_query_response_time
from app.models import User, ChatMessage, Document, ChatSession, LLMUsage, ResponseFeedback
from app.rate_limit import CHAT_QUERY_RATE_LIMIT, check_chat_ws_rate_limit, limiter
from app.rag.utils.security import UnsafePromptError, validate_user_input
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessageResponse,
    ChatHistoryResponse,
    FeedbackRequest,
    SourceChunk,
    GraphContextInfo,
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionResponse,
    QuizPayload,
    QuizStateUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

SOURCE_UNAVAILABLE_MESSAGE = (
    "This source document is unavailable. It may have been moved to trash "
    "or permanently deleted."
)


def _raise_source_unavailable() -> None:
    """Signal that an attached PDF cannot be used for new retrieval."""
    raise AppException(
        "SOURCE_UNAVAILABLE",
        SOURCE_UNAVAILABLE_MESSAGE,
        404,
    )


def _get_active_owned_document(
    db: Session,
    user_id,
    document_id: str,
) -> Optional[Document]:
    """Return an owned, non-trashed document or ``None`` if unavailable."""
    return (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.user_id == user_id,
            Document.is_deleted.is_(False),
        )
        .first()
    )


def _require_active_owned_documents(
    db: Session,
    user_id,
    document_ids: List[str],
) -> List[Document]:
    """Load attached docs; raise if any id is missing or in trash."""
    docs = (
        db.query(Document)
        .filter(
            Document.id.in_(document_ids),
            Document.user_id == user_id,
            Document.is_deleted.is_(False),
        )
        .all()
    )
    found_ids = {str(doc.id) for doc in docs}
    missing = [doc_id for doc_id in document_ids if str(doc_id) not in found_ids]
    if missing:
        _raise_source_unavailable()
    return docs


def _infer_usage_type(usage: Optional[dict], quiz=None) -> str:
    if quiz:
        return "quiz"
    if isinstance(usage, dict):
        explicit = usage.get("usage_type")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip().lower()
    return "chat"


def _record_assistant_usage(
    db: Session,
    *,
    user_id: str,
    usage: Optional[dict],
    session_id: Optional[str] = None,
    message_id=None,
    document_id: Optional[str] = None,
    llm_provider: Optional[str] = None,
    quiz=None,
) -> None:
    """Persist a ledger row for an assistant turn (own-user scoped)."""
    if not usage:
        return
    try:
        from app.pricing import enrich_usage_with_pricing
        from app.services.usage_tracker import record_llm_usage

        provider = llm_provider
        if isinstance(usage, dict) and usage.get("provider"):
            provider = usage.get("provider")
        enriched = enrich_usage_with_pricing(
            usage if isinstance(usage, dict) else {},
            provider=provider,
        )
        record_llm_usage(
            db,
            user_id=str(user_id),
            usage=enriched,
            usage_type=_infer_usage_type(enriched, quiz=quiz),
            provider=enriched.get("provider") or provider,
            model_id=enriched.get("model_id"),
            session_id=session_id,
            message_id=str(message_id) if message_id else None,
            document_id=document_id,
            commit=True,
        )
    except Exception:
        logger.exception("AI usage recording failed for user %s", user_id)


def _parse_message_sources_json(
    raw: Optional[str],
) -> tuple[list, Optional[dict], Optional[str], Optional[int], Optional[dict], list]:
    """Parse sources_json supporting legacy list and enriched object payloads.

    Returns (sources, usage, llm_provider, response_time_ms, quiz, attachments).
    """
    if not raw:
        return [], None, None, None, None, []
    try:
        parsed = json.loads(raw)
    except Exception:
        return [], None, None, None, None, []

    usage = None
    llm_provider = None
    response_time_ms = None
    quiz = None
    attachments: list = []
    chunks = parsed

    if isinstance(parsed, dict):
        chunks = parsed.get("chunks") or parsed.get("sources") or []
        usage = parsed.get("usage")
        llm_provider = parsed.get("llm_provider")
        response_time_ms = parsed.get("response_time_ms")
        quiz = parsed.get("quiz") if isinstance(parsed.get("quiz"), dict) else None
        if usage and isinstance(usage, dict) and not llm_provider:
            llm_provider = usage.get("provider")
        raw_attachments = parsed.get("attachments")
        if isinstance(raw_attachments, list):
            for item in raw_attachments:
                if not isinstance(item, dict):
                    continue
                doc_id = item.get("id")
                filename = item.get("filename") or item.get("original_name")
                if doc_id and filename:
                    attachments.append({"id": str(doc_id), "filename": str(filename)})

    sources: list = []
    if isinstance(chunks, list):
        for item in chunks:
            if not isinstance(item, dict):
                continue
            try:
                sources.append(SourceChunk(**item))
            except Exception:
                continue
    return sources, usage, llm_provider, response_time_ms, quiz, attachments


def _parse_graph_from_sources_json(raw: Optional[str]) -> Optional[dict]:
    """Return the graph metadata stored with a message, if any."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if isinstance(parsed, dict):
        graph = parsed.get("graph")
        return graph if isinstance(graph, dict) else None
    return None


def _coerce_graph_payload(graph: Optional[dict]) -> Optional[GraphContextInfo]:
    if not graph or not isinstance(graph, dict):
        return None
    try:
        return GraphContextInfo(**graph)
    except Exception:
        return None


def _coerce_quiz_payload(quiz: Optional[dict]) -> Optional[QuizPayload]:
    if not quiz or not isinstance(quiz, dict):
        return None
    try:
        return QuizPayload(**quiz)
    except Exception:
        return None


def _message_response_from_row(msg: ChatMessage) -> ChatMessageResponse:
    from app.schemas import AttachedDocument

    sources, usage, llm_provider, response_time_ms, quiz, attachments = (
        _parse_message_sources_json(msg.sources_json)
    )
    return ChatMessageResponse(
        id=str(msg.id),
        role=msg.role,
        content=msg.content,
        sources=sources,
        attachments=[AttachedDocument(**a) for a in attachments],
        feedback=msg.feedback,
        created_at=msg.created_at,
        usage=usage,
        llm_provider=llm_provider,
        response_time_ms=response_time_ms,
        quiz=_coerce_quiz_payload(quiz),
        graph=_coerce_graph_payload(_parse_graph_from_sources_json(msg.sources_json)),
    )


def _enriched_sources_payload(
    sources: list,
    *,
    quiz: Optional[dict] = None,
    usage: Optional[dict] = None,
    llm_provider: Optional[str] = None,
    response_time_ms: Optional[int] = None,
    attachments: Optional[list] = None,
    graph: Optional[dict] = None,
):
    """Build sources_json value: plain list when no metadata, else enriched dict."""
    if (
        not quiz
        and not usage
        and not llm_provider
        and response_time_ms is None
        and not attachments
        and not graph
    ):
        return sources
    enriched_usage = usage
    if isinstance(usage, dict) and usage:
        try:
            from app.pricing import enrich_usage_with_pricing

            enriched_usage = enrich_usage_with_pricing(
                usage, provider=llm_provider or usage.get("provider")
            )
        except Exception:
            enriched_usage = usage
    payload = {
        "chunks": sources if isinstance(sources, list) else [],
        "usage": enriched_usage,
        "llm_provider": llm_provider
        or (
            (enriched_usage or {}).get("provider")
            if isinstance(enriched_usage, dict)
            else None
        ),
        "model_id": (
            (enriched_usage or {}).get("model_id")
            if isinstance(enriched_usage, dict)
            else None
        ),
        "response_time_ms": response_time_ms,
    }
    if quiz:
        payload["quiz"] = quiz
    if attachments:
        payload["attachments"] = attachments
    if graph:
        payload["graph"] = graph
    return payload


@router.websocket("/ws")
async def chat_ws(websocket: WebSocket, token: Optional[str] = Query(None)):
    """WebSocket endpoint for streaming agentic thoughts and tokens.

    Authenticate via the first JSON message containing
    `{token, question, document_id?, session_id?}`.
    Query-string `?token=` is rejected in production (URL/proxy log leakage).
    """
    await websocket.accept()

    # DB-backed auth mirroring get_current_user (incl. token_version)
    from app.database import SessionLocal
    from app.auth import decode_token_with_version
    from app.models import ApiKey, User
    from app.config import get_settings

    db = SessionLocal()
    user = None

    try:
        # Prefer first-message token (keeps JWT out of URL/proxy logs).
        settings = get_settings()
        if token and settings.ENVIRONMENT == "production":
            await websocket.send_json(
                {
                    "type": "error",
                    "data": "Pass the access token in the first WebSocket message, not the URL.",
                }
            )
            await websocket.close()
            return

        if token:
            tok = token
            initial_payload = None
        else:
            msg = await websocket.receive_json()
            tok = msg.get("token")
            initial_payload = msg

        # Prefer first-message / query token; fall back to HttpOnly access_token cookie.
        if not tok:
            tok = websocket.cookies.get("access_token")

        if not tok:
            await websocket.send_json({"type": "error", "data": "Missing token"})
            await websocket.close()
            return

        if tok.startswith("pdf_rag_"):
            import hashlib
            hashed = hashlib.sha256(tok.encode("utf-8")).hexdigest()
            api_key = db.query(ApiKey).filter(ApiKey.hashed_key == hashed, ApiKey.is_active == True).first()
            if not api_key:
                await websocket.send_json({"type": "error", "data": "Invalid API key"})
                await websocket.close()
                return
            user = api_key.user
        else:
            decoded = decode_token_with_version(tok, token_type="access")
            if not decoded:
                await websocket.send_json({"type": "error", "data": "Invalid or expired token"})
                await websocket.close()
                return
            user_id, token_tv = decoded
            user = db.query(User).filter(User.id == user_id).first()
            if user and int(getattr(user, "token_version", 0) or 0) != token_tv:
                await websocket.send_json({"type": "error", "data": "Invalid or expired token"})
                await websocket.close()
                return

        if not user:
            await websocket.send_json({"type": "error", "data": "User not found"})
            await websocket.close()
            return

        # /chat/ask and /chat/ask/stream enforce CHAT_QUERY_RATE_LIMIT via
        # @limiter.limit, but that decorator only works on HTTP routes — it
        # never runs for this WebSocket handler. Without this check, a client
        # could call the same generate_answer_stream(...) RAG/LLM pipeline an
        # unbounded number of times per minute by sending requests over /chat/ws
        # (or opening multiple connections), completely bypassing the limit.
        if not check_chat_ws_rate_limit(user.id):
            await websocket.send_json({"type": "error", "data": "Rate limit exceeded"})
            await websocket.close()
            return

        # Receive or reuse initial payload
        if initial_payload:
            payload = initial_payload
        else:
            payload = await websocket.receive_json()

        question = payload.get("question")
        document_id = payload.get("document_id") or None
        document_ids = payload.get("document_ids") or None
        if isinstance(document_ids, list):
            document_ids = [str(d) for d in document_ids if d]
            if not document_ids:
                document_ids = None
        else:
            document_ids = None
        # Prefer explicit multi-attach list over a single document_id.
        if document_ids:
            document_id = None
        session_id = payload.get("session_id")
        llm_provider = payload.get("llm_provider")
        attachments_meta = payload.get("attachments") if isinstance(payload.get("attachments"), list) else None

        from app.rag.utils.security import validate_user_input, UnsafePromptError

        try:
            validate_user_input(question)
        except UnsafePromptError as exc:
            await websocket.send_json({"type": "error", "data": str(exc)})
            await websocket.close()
            return

        # Validate document if given
        if document_id:
            doc = _get_active_owned_document(db, user.id, document_id)
            if not doc:
                await websocket.send_json(
                    {"type": "error", "data": SOURCE_UNAVAILABLE_MESSAGE}
                )
                await websocket.close()
                return
            if doc.status != "ready":
                progress = getattr(doc, "processing_progress", None)
                stage = getattr(doc, "processing_stage", None)
                detail = f"Document is still {doc.status}."
                if progress is not None:
                    detail += f" Progress: {progress}%"
                if stage:
                    detail += f" Stage: {stage}"
                await websocket.send_json({"type": "error", "data": detail})
                await websocket.close()
                return
        elif document_ids:
            try:
                docs = _require_active_owned_documents(db, user.id, document_ids)
            except AppException as exc:
                await websocket.send_json({"type": "error", "data": str(exc)})
                await websocket.close()
                return
            not_ready = [doc.original_name for doc in docs if doc.status != "ready"]
            if not_ready:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": f"Some documents are still processing: {', '.join(not_ready)}",
                    }
                )
                await websocket.close()
                return
            if not attachments_meta:
                attachments_meta = [
                    {"id": str(doc.id), "filename": doc.original_name} for doc in docs
                ]

        # Resolve owned session (validates foreign session IDs)
        try:
            session = _resolve_owned_session(db, user.id, session_id)
        except NotFoundException as exc:
            await websocket.send_json({"type": "error", "data": str(exc)})
            await websocket.close()
            return
        session_id = session.id

        chat_history = _load_prompt_chat_history(db, user.id, session_id)

        # Save user message (attachments stored in sources_json for history chips)
        user_sources = None
        if attachments_meta:
            user_sources = _enriched_sources_payload([], attachments=attachments_meta)
        save_doc_id = document_id or (document_ids[0] if document_ids else None)
        try:
            _save_message(
                db,
                user.id,
                save_doc_id,
                "user",
                question,
                sources=user_sources,
                session_id=session_id,
            )
        except Exception as e:
            await websocket.send_json({"type": "error", "data": f"Failed to persist message: {e}"})
            await websocket.close()
            return

        # Stream answer using existing generator and forward structured events
        try:
            last_usage = None
            full_answer = ""
            sources: list = []
            quiz_payload = None
            graph_payload = None
            stream_completed = False
            stream_failed = False
            for chunk in generate_answer_stream(
                question=question,
                user_id=user.id,
                document_id=document_id,
                document_ids=document_ids,
                chat_history=chat_history,
                llm_provider=llm_provider,
            ):
                # chunk is SSE-style string like 'data: {json}\n\n' or similar
                try:
                    if chunk.startswith("data: "):
                        event_payload = json.loads(chunk[6:].strip())
                        event_type = event_payload.get("type")
                        if event_type == "usage":
                            last_usage = event_payload.get("data")
                        elif event_type == "error":
                            stream_failed = True
                            stream_completed = False
                        elif event_type == "token":
                            full_answer += event_payload.get("data") or ""
                        elif event_type == "sources":
                            sources = event_payload.get("data") or []
                        elif event_type == "quiz":
                            quiz_payload = event_payload.get("data")
                        elif event_type == "graph":
                            graph_payload = event_payload.get("data")
                        if event_type == "done":
                            if not stream_failed:
                                stream_completed = True
                            continue
                        await websocket.send_json(event_payload)
                    else:
                        # Fallback: send raw token
                        await websocket.send_json({"type": "token", "data": chunk})
                except Exception:
                    await websocket.send_json({"type": "token", "data": chunk})

            assistant_message_id = None
            # Persist only completed non-empty assistant turns (or quiz payloads)
            if (
                stream_completed
                and not stream_failed
                and (full_answer.strip() or quiz_payload)
            ):
                save_sources = _enriched_sources_payload(
                    sources,
                    quiz=quiz_payload,
                    usage=last_usage,
                    llm_provider=llm_provider,
                    graph=graph_payload,
                )
                assistant_message_id = _save_message(
                    db,
                    user.id,
                    document_id,
                    "assistant",
                    full_answer,
                    save_sources,
                    session_id=session_id,
                    allow_empty=bool(quiz_payload),
                )
                if assistant_message_id:
                    _record_assistant_usage(
                        db,
                        user_id=user.id,
                        usage=last_usage,
                        session_id=session_id,
                        message_id=assistant_message_id,
                        document_id=document_id,
                        llm_provider=llm_provider,
                        quiz=quiz_payload,
                    )

            # Notify client (include Bedrock usage when present)
            done_event = {"type": "done"}
            if last_usage:
                done_event["usage"] = last_usage
            if assistant_message_id:
                done_event["message_id"] = str(assistant_message_id)
            if not stream_completed:
                done_event["incomplete"] = True
            await websocket.send_json(done_event)

        except WebSocketDisconnect:
            # Client gone — user message already persisted; do not invent assistant text
            return
        except Exception as e:
            await websocket.send_json({"type": "error", "data": str(e)})

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass
    finally:
        db.close()


@router.get(
    "/sessions",
    response_model=List[ChatSessionResponse],
    summary="List chat sessions",
    description="Returns all chat sessions owned by the authenticated user, newest first.",
)
def get_chat_sessions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve all chat sessions for the authenticated user."""
    sessions = (
        db.query(ChatSession).filter(ChatSession.user_id == user.id).order_by(ChatSession.created_at.desc()).all()
    )
    return sessions


@router.post(
    "/sessions",
    response_model=ChatSessionResponse,
    status_code=201,
    summary="Create a chat session",
    description="Creates a named chat session owned by the authenticated user.",
)
def create_chat_session(
    payload: ChatSessionCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new chat session for the authenticated user."""
    session = ChatSession(
        user_id=user.id,
        title=payload.title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.put(
    "/sessions/{session_id}",
    response_model=ChatSessionResponse,
    summary="Rename a chat session",
    description="Renames one chat session after verifying it belongs to the authenticated user.",
)
def rename_chat_session(
    session_id: str,
    payload: ChatSessionCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rename an existing chat session owned by the authenticated user."""
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user.id,
        )
        .first()
    )
    if not session:
        raise NotFoundException("Chat session")
    session.title = payload.title
    db.commit()
    db.refresh(session)
    return session


@router.patch(
    "/sessions/{session_id}",
    response_model=ChatSessionResponse,
    summary="Update a chat session title",
    description="Partially updates a chat session title after verifying ownership.",
)
def update_chat_session(
    session_id: str,
    payload: ChatSessionUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the title of an existing chat session owned by the authenticated user."""
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user.id,
        )
        .first()
    )
    if not session:
        raise NotFoundException("Chat session")
    session.title = payload.title
    db.commit()
    db.refresh(session)
    return session


@router.delete(
    "/sessions/{session_id}",
    summary="Delete a chat session",
    description=(
        "Deletes one owned chat session and its messages. "
        "Source PDFs are never deleted when a chat session is removed."
    ),
)
def delete_chat_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a chat session owned by the authenticated user.

    Only the session and its messages are removed. Linked documents stay intact.
    Usage ledger rows keep their cost data but drop session/message FKs so
    MySQL does not reject the delete with a foreign-key constraint error.
    """
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user.id,
        )
        .first()
    )
    if not session:
        raise NotFoundException("Chat session")

    message_ids = [
        row[0]
        for row in db.query(ChatMessage.id)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user.id,
        )
        .all()
    ]

    # Detach usage ledger refs before ORM cascade deletes messages/session.
    # Production MySQL FKs on llm_usage.session_id / message_id are RESTRICT
    # unless the SET NULL migration has already run.
    db.query(LLMUsage).filter(LLMUsage.session_id == session_id).update(
        {LLMUsage.session_id: None},
        synchronize_session=False,
    )
    if message_ids:
        db.query(LLMUsage).filter(LLMUsage.message_id.in_(message_ids)).update(
            {LLMUsage.message_id: None},
            synchronize_session=False,
        )

    try:
        db.delete(session)
        db.commit()
    except IntegrityError:
        db.rollback()
        logger.exception(
            "Failed to delete chat session %s for user %s due to FK constraint",
            session_id,
            user.id,
        )
        raise AppException(
            "SESSION_DELETE_FAILED",
            "Unable to delete this chat session because related records could not be cleared.",
            500,
        ) from None
    return Response(status_code=204)


@router.get(
    "/history/session/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Get chat history for a session",
    description="Returns ordered user and assistant messages for one owned chat session.",
)
def get_session_history(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve ordered chat history for a specific owned chat session."""
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user.id,
        )
        .first()
    )
    if not session:
        raise NotFoundException("Chat session")

    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user.id,
        )
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all()
    )

    formatted = [_message_response_from_row(msg) for msg in messages]
    return ChatHistoryResponse(messages=formatted, document_id=None)


def generate_answer(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[list] = None,
    llm_provider: Optional[str] = None,
    session_id: Optional[str] = None,
):
    from app.rag.llm.agent import generate_answer as _generate_answer

    return _generate_answer(
        question=question,
        user_id=user_id,
        document_id=document_id,
        document_ids=document_ids,
        top_k=top_k,
        chat_history=chat_history,
        llm_provider=llm_provider,
        session_id=session_id,
    )


def generate_answer_stream(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[list] = None,
    llm_provider: Optional[str] = None,
    session_id: Optional[str] = None,
):
    from app.rag.llm.agent import generate_answer_stream as _generate_answer_stream

    return _generate_answer_stream(
        question=question,
        user_id=user_id,
        document_id=document_id,
        document_ids=document_ids,
        top_k=top_k,
        chat_history=chat_history,
        llm_provider=llm_provider,
        session_id=session_id,
    )


@router.post(
    "/ask",
    response_model=ChatResponse,
    summary="Ask a RAG question",
    description=(
        "Runs non-streaming retrieval-augmented generation for the authenticated "
        "user, optionally scoped to one ready document."
    ),
)
@limiter.limit(CHAT_QUERY_RATE_LIMIT)
def ask_question(
    payload: ChatRequest,
    request: Request = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Ask a question with RAG retrieval and return the complete answer."""
    started_at = time.perf_counter()
    
    # Bind query parameters to request context variables and state
    if request is not None:
        request.state.query = payload.question
    from app.observability import query_text_var
    query_text_var.set(payload.question)
    logger.info(f"Processing RAG chat query: '{payload.question}'")

    try:
        try:
            validate_user_input(payload.question)
        except UnsafePromptError as exc:
            raise ValidationException(str(exc)) from exc

        # Validate document exists if specified
        if payload.document_id:
            doc = _get_active_owned_document(db, user.id, payload.document_id)
            if not doc:
                _raise_source_unavailable()

            if doc.status != "ready":
                progress = getattr(doc, "processing_progress", None)
                stage = getattr(doc, "processing_stage", None)
                detail = f"Document is still {doc.status}. Please wait for processing to complete."
                if progress is not None:
                    detail += f" Progress: {progress}%"
                if stage:
                    detail += f" Stage: {stage}"
                raise ValidationException(detail)

            # Update last_accessed_at timestamp
            doc.last_accessed_at = datetime.now(timezone.utc)
            db.commit()

        # Validate documents if multiple specified
        elif payload.document_ids:
            docs = _require_active_owned_documents(
                db, user.id, [str(d) for d in payload.document_ids]
            )

            not_ready = [doc.original_name for doc in docs if doc.status != "ready"]
            if not_ready:
                raise ValidationException(
                    f"Some documents are still processing: {', '.join(not_ready)}. Please wait."
                )

            for doc in docs:
                doc.last_accessed_at = datetime.now(timezone.utc)
            db.commit()

        # Resolve owned session (never accept foreign session IDs)
        session = _resolve_owned_session(db, user.id, payload.session_id)
        session_id = session.id

        # DB-backed history for this session only (before current turn)
        chat_history = _load_prompt_chat_history(db, user.id, session_id)

        attach_meta = None
        if payload.document_ids:
            attach_docs = (
                db.query(Document)
                .filter(Document.id.in_(payload.document_ids), Document.user_id == user.id)
                .all()
            )
            attach_meta = _attachments_from_docs(attach_docs)
        elif payload.document_id:
            one = (
                db.query(Document)
                .filter(Document.id == payload.document_id, Document.user_id == user.id)
                .first()
            )
            if one:
                attach_meta = _attachments_from_docs([one])
        user_sources = (
            _enriched_sources_payload([], attachments=attach_meta) if attach_meta else None
        )
        msg_doc_id = _message_document_fk(payload.document_id, payload.document_ids)

        # Persist user turn first so context survives generator/LLM failures
        try:
            _save_message(
                db,
                user.id,
                msg_doc_id,
                "user",
                payload.question,
                sources=user_sources,
                session_id=session_id,
            )
        except AppException:
            raise
        except Exception as exc:
            logger.exception("Failed to persist user message for session %s", session_id)
            raise AppException(
                "DATABASE_ERROR",
                "Failed to persist chat message. The message was not saved.",
                500,
                {"error": str(exc)[:200]},
            ) from exc

        cache_doc_key = str(payload.document_id or "")
        if payload.document_ids:
            cache_doc_key = "multi:" + ",".join(sorted(payload.document_ids))
        cache_question = payload.question
        if payload.llm_provider:
            cache_question = f"[{payload.llm_provider}] {payload.question}"
        # Never cache quiz intents — failed/partial quiz answers must not sticky-cache.
        from app.rag.utils.quiz import is_quiz_request

        quiz_intent_request = is_quiz_request(payload.question)
        cached_payload = None
        if not quiz_intent_request:
            cached_payload = get_cached_payload(
                user_id=user.id,
                document_id=cache_doc_key,
                question=cache_question,
                session_id=str(session_id),
            )

        if cached_payload is not None:
            logger.debug("Returning cached response for question: %s", payload.question[:40])
            answer = cached_payload.get("answer") or ""
            raw_sources = cached_payload.get("sources") or []
            save_sources = _enriched_sources_payload(raw_sources)
            if answer.strip():
                _save_message(
                    db,
                    user.id,
                    msg_doc_id,
                    "assistant",
                    answer,
                    save_sources,
                    session_id=session_id,
                )
            cached_sources = []
            for item in raw_sources:
                if isinstance(item, dict):
                    try:
                        cached_sources.append(SourceChunk(**item))
                    except Exception:
                        continue
            return ChatResponse(
                answer=answer,
                sources=cached_sources,
                document_id=payload.document_id,
            )

        result = generate_answer(
            question=payload.question,
            user_id=user.id,
            document_id=payload.document_id,
            document_ids=payload.document_ids,
            top_k=payload.top_k,
            chat_history=chat_history,
            llm_provider=payload.llm_provider,
            session_id=str(session_id),
        )

        # Bind chunks retrieved to request state and context variables
        chunks_count = len(result.get("sources", []))
        if request is not None:
            request.state.chunks_retrieved = chunks_count
        from app.observability import chunks_retrieved_var
        chunks_retrieved_var.set(chunks_count)
        logger.info(f"RAG chat query processed successfully, retrieved {chunks_count} chunks")

        # Store result in cache for future identical questions (skip interactive quizzes)
        quiz_data = result.get("quiz")
        if not quiz_data and not quiz_intent_request and (result.get("answer") or "").strip():
            set_cached_response(
                user_id=user.id,
                document_id=cache_doc_key,
                question=cache_question,
                answer=result["answer"],
                sources=result.get("sources") or [],
                session_id=str(session_id),
            )

        save_sources = _enriched_sources_payload(
            result.get("sources") or [],
            quiz=quiz_data,
            usage=result.get("usage"),
            llm_provider=payload.llm_provider,
            graph=result.get("graph"),
        )
        assistant_message_id = _save_message(
            db,
            user.id,
            msg_doc_id,
            "assistant",
            result.get("answer") or "",
            save_sources,
            session_id=session_id,
            allow_empty=bool(quiz_data),
        )
        if assistant_message_id:
            _record_assistant_usage(
                db,
                user_id=user.id,
                usage=result.get("usage"),
                session_id=session_id,
                message_id=assistant_message_id,
                document_id=msg_doc_id,
                llm_provider=payload.llm_provider,
                quiz=quiz_data,
            )

        safe_sources = []
        for s in result.get("sources") or []:
            try:
                safe_sources.append(SourceChunk(**s))
            except Exception:
                continue

        return ChatResponse(
            answer=result["answer"],
            sources=safe_sources,
            document_id=payload.document_id,
            quiz=_coerce_quiz_payload(quiz_data),
            graph=_coerce_graph_payload(result.get("graph")),
        )
    finally:
        record_query_response_time(time.perf_counter() - started_at)


@router.post(
    "/ask/stream",
    summary="Stream a RAG answer",
    description=(
        "Runs retrieval-augmented generation and streams answer tokens as "
        "server-sent events. The final assistant response is saved to history."
    ),
)
@limiter.limit(CHAT_QUERY_RATE_LIMIT)
def ask_question_stream(
    payload: ChatRequest,
    request: Request = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Ask a question and stream the answer using Server-Sent Events."""
    # Bind query parameters to request context variables and state
    if request is not None:
        request.state.query = payload.question
    from app.observability import query_text_var
    query_text_var.set(payload.question)
    logger.info(f"Processing streaming RAG chat query: '{payload.question}'")

    try:
        validate_user_input(payload.question)
    except UnsafePromptError as exc:
        raise ValidationException(str(exc)) from exc

    # Validate document
    if payload.document_id:
        doc = _get_active_owned_document(db, user.id, payload.document_id)
        if not doc:
            _raise_source_unavailable()

        if doc.status != "ready":
            progress = getattr(doc, "processing_progress", None)
            stage = getattr(doc, "processing_stage", None)
            detail = f"Document is still {doc.status}. Please wait for processing to complete."
            if progress is not None:
                detail += f" Progress: {progress}%"
            if stage:
                detail += f" Stage: {stage}"
            raise ValidationException(detail)

        # Update last_accessed_at timestamp
        doc.last_accessed_at = datetime.now(timezone.utc)
        db.commit()

    # Validate documents if multiple specified
    elif payload.document_ids:
        docs = _require_active_owned_documents(
            db, user.id, [str(d) for d in payload.document_ids]
        )

        not_ready = [doc.original_name for doc in docs if doc.status != "ready"]
        if not_ready:
            raise ValidationException(
                f"Some documents are still processing: {', '.join(not_ready)}. Please wait."
            )

        for doc in docs:
            doc.last_accessed_at = datetime.now(timezone.utc)
        db.commit()

    started_at = time.perf_counter()

    session = _resolve_owned_session(db, user.id, payload.session_id)
    session_id = session.id

    # History before current user turn
    chat_history = _load_prompt_chat_history(db, user.id, session_id)

    cache_doc_key = str(payload.document_id or "")
    if payload.document_ids:
        cache_doc_key = "multi:" + ",".join(sorted(payload.document_ids))

    attach_meta = None
    if payload.document_ids:
        attach_docs = (
            db.query(Document)
            .filter(Document.id.in_(payload.document_ids), Document.user_id == user.id)
            .all()
        )
        attach_meta = _attachments_from_docs(attach_docs)
    elif payload.document_id:
        one = (
            db.query(Document)
            .filter(Document.id == payload.document_id, Document.user_id == user.id)
            .first()
        )
        if one:
            attach_meta = _attachments_from_docs([one])
    user_sources = (
        _enriched_sources_payload([], attachments=attach_meta) if attach_meta else None
    )
    msg_doc_id = _message_document_fk(payload.document_id, payload.document_ids)

    # Save user message immediately (survives stream failures / disconnects)
    try:
        _save_message(
            db,
            user.id,
            msg_doc_id,
            "user",
            payload.question,
            sources=user_sources,
            session_id=session_id,
        )
    except Exception:
        logger.exception("Failed to persist user message for stream session %s", session_id)
        raise

    # Cache check before starting the stream (skip quiz intents entirely)
    from app.rag.utils.quiz import is_quiz_request as _is_quiz_request

    stream_quiz_intent = _is_quiz_request(payload.question)
    stream_cache_question = payload.question
    if payload.llm_provider:
        stream_cache_question = f"[{payload.llm_provider}] {payload.question}"
    if not stream_quiz_intent:
        cached_payload = get_cached_payload(
            user_id=user.id,
            document_id=cache_doc_key,
            question=stream_cache_question,
            session_id=str(session_id),
        )
        if cached_payload is not None:
            logger.debug("Returning cached stream response for question: %s", payload.question[:40])
            cached_answer = cached_payload.get("answer") or ""
            cached_sources = cached_payload.get("sources") or []
            cached_message_id = None
            if cached_answer.strip():
                cached_message_id = _save_message(
                    db,
                    user.id,
                    msg_doc_id,
                    "assistant",
                    cached_answer,
                    _enriched_sources_payload(cached_sources),
                    session_id=session_id,
                )

            async def cached_event_stream():
                # Emit sources first so citation chips work on cache hits.
                if cached_sources:
                    yield f"data: {json.dumps({'type': 'sources', 'data': cached_sources})}\n\n"
                # Stream cached text in small pieces so the UI can paint
                # progressively (ChatGPT-style) instead of dumping one blob.
                chunk_size = 24
                for i in range(0, len(cached_answer), chunk_size):
                    piece = cached_answer[i : i + chunk_size]
                    if piece:
                        yield f"data: {json.dumps({'type': 'token', 'data': piece})}\n\n"
                        await asyncio.sleep(0.012)
                done_payload = {"type": "done"}
                if cached_message_id:
                    done_payload["message_id"] = str(cached_message_id)
                yield f"data: {json.dumps(done_payload)}\n\n"

            return StreamingResponse(
                cached_event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    # Stream response
    def event_stream():
        full_answer = ""
        sources = []
        quiz_payload = None
        graph_payload = None
        last_usage = None
        stream_completed = False
        stream_failed = False
        assistant_persisted = False

        try:
            for chunk in generate_answer_stream(
                question=payload.question,
                user_id=user.id,
                document_id=None if payload.document_ids else payload.document_id,
                document_ids=payload.document_ids,
                top_k=payload.top_k,
                chat_history=chat_history,
                llm_provider=payload.llm_provider,
                session_id=str(session_id),
            ):
                # Parse to accumulate full answer for history
                try:
                    if chunk.startswith("data: "):
                        data = json.loads(chunk[6:].strip())
                        event_type = data.get("type")
                        if event_type == "usage":
                            last_usage = data.get("data")
                        elif event_type == "error":
                            stream_failed = True
                            stream_completed = False
                        elif event_type == "done":
                            if not stream_failed:
                                stream_completed = True
                            # Capture usage from done payload before replacing with timed done.
                            if data.get("usage"):
                                last_usage = data.get("usage")
                            continue  # We will yield our own done event with response time
                        if event_type == "token":
                            full_answer += data.get("data", "")
                        elif event_type == "sources":
                            sources = data.get("data", [])
                        elif event_type == "quiz":
                            quiz_payload = data.get("data")
                        elif event_type == "graph":
                            graph_payload = data.get("data")
                except Exception:
                    pass
                yield chunk

            # Only persist a completed stream — interrupted/failed streams skip assistant save
            if stream_failed or not stream_completed:
                logger.warning(
                    "Stream ended without completion for session %s; skipping assistant persist",
                    session_id,
                )
                elapsed_ms = round((time.perf_counter() - started_at) * 1000)
                yield f"data: {json.dumps({'type': 'done', 'response_time_ms': elapsed_ms, 'incomplete': True})}\n\n"
                return

            # Cache the full answer for future identical questions
            # Include provider in cache key so Ollama/Bedrock answers stay separate.
            # Skip quiz intents entirely (success or failure) so retries stay live.
            if full_answer.strip() and not quiz_payload and not stream_quiz_intent:
                set_cached_response(
                    user_id=user.id,
                    document_id=cache_doc_key,
                    question=stream_cache_question,
                    answer=full_answer,
                    sources=sources or [],
                    session_id=str(session_id),
                )

            # Save assistant response to history (include usage metadata when present)
            from app.database import get_db_session

            elapsed_ms = round((time.perf_counter() - started_at) * 1000)
            save_sources = _enriched_sources_payload(
                sources,
                quiz=quiz_payload,
                usage=last_usage,
                llm_provider=payload.llm_provider,
                response_time_ms=elapsed_ms,
                graph=graph_payload,
            )

            assistant_message_id = None
            if (full_answer.strip() or quiz_payload) and not assistant_persisted:
                try:
                    with get_db_session() as save_db:
                        assistant_message_id = _save_message(
                            save_db,
                            user.id,
                            msg_doc_id,
                            "assistant",
                            full_answer,
                            save_sources,
                            session_id=session_id,
                            allow_empty=bool(quiz_payload),
                        )
                        if assistant_message_id:
                            assistant_persisted = True
                            _record_assistant_usage(
                                save_db,
                                user_id=user.id,
                                usage=last_usage,
                                session_id=session_id,
                                message_id=assistant_message_id,
                                document_id=msg_doc_id,
                                llm_provider=payload.llm_provider,
                                quiz=quiz_payload,
                            )
                except Exception:
                    logger.exception(
                        "Failed to persist assistant stream message for session %s",
                        session_id,
                    )
                    yield f"data: {json.dumps({'type': 'error', 'data': 'Failed to persist assistant response'})}\n\n"
                    return

            # Log streaming response RAG completion
            chunks_count = len(sources)
            from app.observability import chunks_retrieved_var, query_text_var, user_id_var
            user_id_var.set(user.id)
            query_text_var.set(payload.question)
            chunks_retrieved_var.set(chunks_count)
            logger.info(f"Streaming RAG chat query completed, retrieved {chunks_count} chunks")

            done_payload = {"type": "done", "response_time_ms": elapsed_ms}
            if last_usage:
                done_payload["usage"] = last_usage
            if assistant_message_id:
                done_payload["message_id"] = str(assistant_message_id)
            yield f"data: {json.dumps(done_payload)}\n\n"
        except Exception as exc:
            logger.exception("Streaming RAG failed for session %s: %s", session_id, exc)
            # Do not persist a misleading completed assistant message on failure
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"
        finally:
            record_query_response_time(time.perf_counter() - started_at)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/history/{document_id}",
    response_model=ChatHistoryResponse,
    summary="Get document chat history",
    description="Returns ordered chat messages for one document owned by the authenticated user.",
)
def get_chat_history(
    document_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve the complete chat history for a specific document."""
    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.user_id == user.id,
            ChatMessage.document_id == document_id,
        )
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    formatted = [_message_response_from_row(msg) for msg in messages]
    return ChatHistoryResponse(messages=formatted, document_id=document_id)


def _export_response(title: str, messages, format: str) -> Response:
    """Build a downloadable chat export Response for md/txt/pdf."""
    from types import SimpleNamespace

    if format not in ("md", "txt", "pdf"):
        raise ValidationException("Format must be 'md', 'txt', or 'pdf'")
    if not messages:
        raise NotFoundException("Chat history")

    title_doc = SimpleNamespace(original_name=title)
    if format == "md":
        content = _format_markdown(title_doc, messages)
        media_type = "text/markdown"
        extension = "md"
    elif format == "txt":
        content = _format_plaintext(title_doc, messages)
        media_type = "text/plain"
        extension = "txt"
    else:
        from app.routes.chat_export import format_pdf as _format_pdf

        content = _format_pdf(title_doc, messages)
        media_type = "application/pdf"
        extension = "pdf"

    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title).strip() or "chat"
    safe_name = safe_name.replace(" ", "_")[:80]
    filename = f"{safe_name}_chat_history.{extension}"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get(
    "/export/session/{session_id}",
    summary="Export session chat history",
    description=(
        "Downloads one chat session as Markdown, plain text, or PDF. "
        "Prefer Authorization: Bearer. Query ?token= is deprecated."
    ),
)
def export_session_chat_history(
    session_id: str,
    format: str = "md",
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Export chat history for a session as a downloadable file."""
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user.id,
        )
        .first()
    )
    if not session:
        raise NotFoundException("Chat session")

    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.user_id == user.id,
            ChatMessage.session_id == session_id,
        )
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return _export_response(session.title or "Chat", messages, format)


@router.get(
    "/export/{document_id}",
    summary="Export document chat history",
    description=(
        "Downloads one document's chat history as Markdown, plain text, or PDF. "
        "Prefer Authorization: Bearer. Query ?token= is deprecated."
    ),
)
def export_chat_history(
    document_id: str,
    format: str = "md",
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Export the chat history for a document as a downloadable file.

    Soft-deleted (trashed) documents remain exportable so conversation history
    is still available from Trash until permanent purge.
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == user.id,
    ).first()

    if not doc:
        raise NotFoundException("Document")

    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.user_id == user.id,
            ChatMessage.document_id == document_id,
        )
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return _export_response(doc.original_name, messages, format)


@router.delete(
    "/history/{document_id}",
    summary="Clear document chat history",
    description="Deletes all chat messages for one document owned by the authenticated user.",
)
def clear_chat_history(
    document_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete all chat messages associated with a specific document."""
    # Delete the ChatMessage records
    db.query(ChatMessage).filter(
        ChatMessage.user_id == user.id,
        ChatMessage.document_id == document_id,
    ).delete(synchronize_session=False)
    db.commit()

    return {"message": "Chat history cleared"}


# Predefined reason categories accepted from the chat UI. A response can be
# rated with a category, a free-text comment, or both.
_FEEDBACK_REASONS = {
    "up": {"accurate", "helpful", "clear", "well_cited", "other"},
    "down": {
        "inaccurate",
        "incomplete",
        "not_relevant",
        "unclear",
        "poor_citation",
        "too_long",
        "other",
    },
}


@router.patch("/feedback/{message_id}")
def submit_feedback(
    message_id: str,
    payload: FeedbackRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit Like/Dislike feedback for one assistant response.

    Stores a snapshot of the user's question and the assistant's complete
    response alongside the rating. A response can be rated only once.
    """
    msg = db.query(ChatMessage).filter(
        ChatMessage.id == message_id,
        ChatMessage.user_id == user.id,
    ).first()

    if not msg:
        raise NotFoundException("Message")
    if msg.role != "assistant":
        raise ValidationException("Can only provide feedback on assistant messages")
    if not (msg.content or "").strip():
        raise ValidationException("Cannot rate an empty response")

    if payload.reason:
        allowed = _FEEDBACK_REASONS[payload.feedback]
        if payload.reason not in allowed:
            raise ValidationException(
                f"Unknown feedback reason '{payload.reason}'"
            )

    existing = (
        db.query(ResponseFeedback)
        .filter(ResponseFeedback.message_id == msg.id)
        .first()
    )
    if existing:
        raise AppException(
            "FEEDBACK_EXISTS",
            "Feedback has already been submitted for this response",
            409,
        )

    # Find the user's prompt that produced this response. Prefer the same
    # conversation so the feedback stays attached to the exact turn.
    question_query = db.query(ChatMessage).filter(
        ChatMessage.user_id == msg.user_id,
        ChatMessage.role == "user",
        ChatMessage.created_at <= msg.created_at,
    )
    if msg.session_id:
        question_query = question_query.filter(
            ChatMessage.session_id == msg.session_id
        )
    elif msg.document_id:
        question_query = question_query.filter(
            ChatMessage.document_id == msg.document_id
        )
    question_msg = question_query.order_by(
        ChatMessage.created_at.desc(), ChatMessage.id.desc()
    ).first()

    feedback = ResponseFeedback(
        user_id=msg.user_id,
        session_id=msg.session_id,
        message_id=msg.id,
        question_message_id=question_msg.id if question_msg else None,
        question=question_msg.content if question_msg else "",
        response=msg.content or "",
        feedback=payload.feedback,
        reason=payload.reason,
        comment=payload.comment,
    )
    msg.feedback = payload.feedback
    db.add(feedback)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise AppException(
            "FEEDBACK_EXISTS",
            "Feedback has already been submitted for this response",
            409,
        ) from exc
    db.refresh(msg)

    return _message_response_from_row(msg)


@router.patch(
    "/messages/{message_id}/quiz-state",
    response_model=ChatMessageResponse,
    summary="Update interactive quiz progress",
    description=(
        "Persists selected answers, current question index, and completion "
        "for an assistant message that contains an interactive quiz."
    ),
)
def update_quiz_state(
    message_id: str,
    payload: QuizStateUpdateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Persist in-chat quiz progress without altering question content."""
    msg = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.id == message_id,
            ChatMessage.user_id == user.id,
        )
        .first()
    )
    if not msg:
        raise NotFoundException("Message")
    if msg.role != "assistant":
        raise ValidationException("Quiz state can only be updated on assistant messages")

    sources, usage, llm_provider, response_time_ms, quiz, _attachments = (
        _parse_message_sources_json(msg.sources_json)
    )
    if not quiz:
        raise ValidationException("Message does not contain a quiz")

    state = quiz.get("state") if isinstance(quiz.get("state"), dict) else {}
    if payload.current_index is not None:
        state["current_index"] = payload.current_index
    if payload.answers is not None:
        existing = state.get("answers") if isinstance(state.get("answers"), dict) else {}
        merged = {**existing, **payload.answers}
        state["answers"] = {str(k): str(v) for k, v in merged.items()}
    if payload.completed is not None:
        state["completed"] = bool(payload.completed)
    quiz["state"] = state

    chunk_dicts = []
    for item in sources:
        if hasattr(item, "model_dump"):
            chunk_dicts.append(item.model_dump())
        elif isinstance(item, dict):
            chunk_dicts.append(item)

    msg.sources_json = json.dumps(
        _enriched_sources_payload(
            chunk_dicts,
            quiz=quiz,
            usage=usage,
            llm_provider=llm_provider,
            response_time_ms=response_time_ms,
        )
    )
    db.commit()
    db.refresh(msg)
    return _message_response_from_row(msg)


def _message_document_fk(
    document_id: Optional[str],
    document_ids: Optional[List[str]] = None,
) -> Optional[str]:
    """Pick a real document UUID for ChatMessage.document_id (never cache keys)."""
    if document_ids:
        return str(document_ids[0])
    if document_id and not str(document_id).startswith("multi:"):
        return str(document_id)
    return None


def _attachments_from_docs(docs: list) -> list:
    return [{"id": str(doc.id), "filename": doc.original_name} for doc in docs]


def _resolve_owned_session(
    db: Session,
    user_id: str,
    session_id: Optional[str] = None,
) -> ChatSession:
    """Resolve a session owned by the user (DB is source of truth)."""
    from app.services.conversation_memory import resolve_session

    try:
        return resolve_session(
            db,
            user_id=user_id,
            session_id=session_id,
            create_if_missing=True,
        )
    except ValueError as exc:
        raise NotFoundException("Chat session") from exc


def _load_prompt_chat_history(
    db: Session,
    user_id: str,
    session_id: str,
) -> list:
    """Load DB-backed history dicts for the agent prompt (excludes current turn)."""
    from app.services.conversation_memory import load_prompt_history

    dicts, _ = load_prompt_history(
        db, user_id=str(user_id), session_id=str(session_id)
    )
    return dicts


def _save_message(
    db: Session,
    user_id: str,
    document_id: Optional[str],
    role: str,
    content: str,
    sources: list = None,
    session_id: Optional[str] = None,
    *,
    allow_empty: bool = False,
):
    """Save a chat message to the database. Returns the new message id or None."""
    from app.services.conversation_memory import persist_chat_message

    if not session_id:
        session = _resolve_owned_session(db, user_id, None)
        session_id = session.id

    msg = persist_chat_message(
        db,
        user_id=str(user_id),
        session_id=str(session_id),
        role=role,
        content=content,
        document_id=document_id,
        sources=sources,
        allow_empty=allow_empty,
    )
    return msg.id if msg is not None else None


def _format_markdown(doc, messages) -> str:
    """Format chat history as a Markdown document."""
    lines = [
        f"# Chat History — {doc.original_name}",
        "",
        f"**Document:** {doc.original_name}  ",
        f"**Exported at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total messages:** {len(messages)}",
        "",
        "---",
        "",
    ]

    for msg in messages:
        timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S") if msg.created_at else ""
        role_label = "**You**" if msg.role == "user" else "**Assistant**"

        lines.append(f"### {role_label}")
        lines.append(f"*{timestamp}*")
        lines.append("")
        lines.append(msg.content)
        lines.append("")

        if msg.role == "assistant" and msg.sources_json:
            try:
                sources = json.loads(msg.sources_json)
                if sources:
                    lines.append("**Sources:**")
                    lines.append("")
                    for i, src in enumerate(sources, 1):
                        lines.append(
                            f"> **[{i}]** {src.get('filename', 'Unknown')}, "
                            f"Page {src.get('page', '?')} "
                            f"(Confidence: {src.get('confidence', 0)}%)"
                        )
                        text_preview = src.get("text", "")[:150]
                        if text_preview:
                            lines.append(f"> {text_preview}...")
                        lines.append(">")
                    lines.append("")
            except Exception:
                pass

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _format_plaintext(doc, messages) -> str:
    """Format chat history as a plain text document."""
    lines = [
        f"Chat History — {doc.original_name}",
        f"Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total messages: {len(messages)}",
        "=" * 60,
        "",
    ]

    for msg in messages:
        timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S") if msg.created_at else ""
        role_label = "You" if msg.role == "user" else "Assistant"

        lines.append(f"[{role_label}] ({timestamp})")
        lines.append(msg.content)

        if msg.role == "assistant" and msg.sources_json:
            try:
                sources = json.loads(msg.sources_json)
                if sources:
                    lines.append("")
                    lines.append("Sources:")
                    for i, src in enumerate(sources, 1):
                        lines.append(
                            f"  [{i}] {src.get('filename', 'Unknown')}, "
                            f"Page {src.get('page', '?')} "
                            f"(Confidence: {src.get('confidence', 0)}%)"
                        )
            except Exception:
                pass

        lines.append("-" * 60)
        lines.append("")

    return "\n".join(lines)