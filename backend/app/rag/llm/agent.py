"""
Agentic RAG — answer generation on top of hybrid retrieval.

LLM backends (verified defaults in config.py):
- ollama (default): LLM_MODEL=qwen2.5:3b, LLM_TEMPERATURE=0.3, LLM_MAX_NEW_TOKENS=1024
- huggingface: ReAct agent with PDFSearchTool / MathTool / WebSearchTool

Default ask path when ``settings.uses_ollama_llm`` is True:
  generate_answer / generate_answer_stream
    → _generate_answer_direct / _generate_answer_stream_direct
    → retrieve(...) → _format_context → SYSTEM_PROMPT + RAG_PROMPT_TEMPLATE
    → OllamaChatClient.chat_completion(model=qwen2.5:3b)

Common example question (all six files):
  "Does the XOLAIR prescribing label recommend every-2-week dosing for
   chronic spontaneous urticaria (CSU)?"
"""
import logging
import json
from typing import List, Dict, Any, Optional, Generator, Tuple

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from app.config import get_settings
from app.rag.retrieval.retriever import retrieve
from app.rag.prompts.prompts import (
    AGENT_SYSTEM_PROMPT,
    GREETING_PROMPT,
    MULTI_DOC_COMPARISON_GUIDANCE,
    RAG_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)


def _resolve_runtime_prompts(
    *,
    question: str,
    user_id: str,
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    retrieved_context: Optional[str] = None,
    llm_provider: Optional[str] = None,
    prompt_type: str = "rag",
):
    """Assemble Global Base + Domain Overlay + Task prompt from DB.

    Falls back to hardcoded ``prompts.py`` constants if resolution fails.
    """
    from app.database import SessionLocal
    from app.services.prompt_management.resolver import (
        ResolvedPrompts,
        infer_task_type,
        resolve_prompts,
    )

    attach_ids = _unique_document_ids(document_ids)
    if document_id and document_id not in attach_ids:
        attach_ids = [document_id, *attach_ids]
    task = prompt_type if prompt_type != "rag" else infer_task_type(question, "rag")
    db = SessionLocal()
    try:
        resolved = resolve_prompts(
            db,
            question=question,
            prompt_type=task,
            user_id=user_id,
            session_id=session_id,
            document_ids=attach_ids,
            retrieved_context=retrieved_context,
            chat_history=chat_history,
            llm_provider=llm_provider,
            persist_execution=True,
        )
        db.commit()
        return resolved
    except Exception as exc:
        logger.warning("Dynamic prompt resolution failed; using static prompts: %s", exc)
        db.rollback()
        return ResolvedPrompts(
            domain="general",
            confidence=0.0,
            was_fallback=True,
            system_prompt=SYSTEM_PROMPT,
            task_prompt=(
                GREETING_PROMPT
                if task == "greeting"
                else AGENT_SYSTEM_PROMPT
                if task == "agent"
                else RAG_PROMPT_TEMPLATE
            ),
            prompt_type=task,
            rationale=f"static fallback: {exc}",
        )
    finally:
        db.close()
from app.exceptions import ExternalServiceException
from app.rag.utils.security import MALFORMED_OUTPUT_MESSAGE, OutputParserError, parse_agent_output
from app.rag.utils.tools import PDFSearchTool, MathTool, WebSearchTool
from app.rag.utils.tracing import trace_function
from app.rag.utils.keywords import extract_keywords

logger = logging.getLogger(__name__)
settings = get_settings()


def persist_document_keywords(document, chunks, db) -> None:
    """
    What: Compute TF-IDF keywords for a document and save them on the DB row.
    Why:  UI / search hints can show key terms without re-scanning all chunks.
    How:  ``extract_keywords`` on chunk texts → ``document.keywords`` JSON string.
    In:   SQLAlchemy document, list of chunk dicts, db session
    Out:  None (mutates document and stages it with ``db.add``)

    Connects to: indexing pipeline after chunks exist (not the ask path).
    Never overwrites existing keywords with an empty extraction result.
    """
    raw_texts: List[str] = []
    for chunk in chunks or []:
        if isinstance(chunk, dict):
            text = (chunk.get("text") or chunk.get("content") or "").strip()
        elif isinstance(chunk, str):
            text = chunk.strip()
        else:
            continue
        if text:
            raw_texts.append(text)

    if not raw_texts:
        logger.info(
            "Skipping keyword persist for document %s — no usable chunk text",
            getattr(document, "id", None),
        )
        return

    kws = extract_keywords(raw_texts, top_n=10)
    if not kws:
        logger.info(
            "Skipping keyword persist for document %s — empty extraction",
            getattr(document, "id", None),
        )
        return

    document.keywords = json.dumps(kws)
    db.add(document)


def normalize_llm_provider(provider: Optional[str] = None) -> str:
    """Resolve per-request or configured LLM provider to a canonical id.

    Production never selects Ollama — requests are coerced to Bedrock.
    """
    chosen = (provider or settings.LLM_PROVIDER or "ollama").strip().lower()
    aliases = {
        "bedrock_nova": "bedrock",
        "nova": "bedrock",
        "nova2": "bedrock",
        "nova_2_lite": "bedrock",
        "qwen": "ollama",
    }
    chosen = aliases.get(chosen, chosen)
    if settings.is_production and chosen == "ollama":
        logger.warning(
            "Ollama LLM requested in production; using Bedrock Nova instead"
        )
        return "bedrock"
    return chosen


def llm_model_for_provider(provider: Optional[str] = None) -> str:
    """Model id passed to chat_completion for the selected provider."""
    chosen = normalize_llm_provider(provider)
    if chosen == "bedrock":
        return settings.BEDROCK_MODEL_ID
    return settings.LLM_MODEL


def get_llm_client(llm_provider: Optional[str] = None):
    """Return InferenceClient-compatible chat client for the active LLM provider.

    Exact default branch
    --------------------
    settings.uses_ollama_llm True:
      check_ollama_reachable("http://localhost:11434")
      check_ollama_model(..., "qwen2.5:3b")
      return OllamaChatClient(base_url=..., model="qwen2.5:3b")

    Bedrock (LLM_PROVIDER=bedrock or llm_provider=bedrock):
      return BedrockNovaChatClient(model_id=us.amazon.nova-2-lite-v1:0)

    Exact call site in direct RAG
    -----------------------------
    client.chat_completion(
      messages=[{"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": user_prompt}],
      model=settings.LLM_MODEL,          # "qwen2.5:3b"
      max_tokens=settings.LLM_MAX_NEW_TOKENS,  # 1024
      temperature=settings.LLM_TEMPERATURE,    # 0.3
      stream=False|True,
    )

    Also used by: retriever._generate_query_variants (rewrite).
    """
    provider = normalize_llm_provider(llm_provider)

    if provider == "bedrock":
        from app.rag.llm.bedrock_client import BedrockNovaChatClient

        return BedrockNovaChatClient(
            model_id=settings.BEDROCK_MODEL_ID,
            region=settings.BEDROCK_REGION,
        )

    if provider == "ollama" or (provider != "huggingface" and settings.uses_ollama_llm):
        from app.rag.llm.ollama_client import (
            OllamaChatClient,
            check_ollama_model,
            check_ollama_reachable,
        )

        ollama_model = llm_model_for_provider("ollama")
        check_ollama_reachable(settings.OLLAMA_BASE_URL)
        check_ollama_model(settings.OLLAMA_BASE_URL, ollama_model)
        return OllamaChatClient(
            base_url=settings.OLLAMA_BASE_URL,
            model=ollama_model,
        )

    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError(
            "LLM_PROVIDER=huggingface requires huggingface-hub. "
            "Install with: pip install -r backend/requirements-dev.txt "
            "(production uses LLM_PROVIDER=bedrock)."
        ) from exc

    return InferenceClient()


def _format_chat_history(
    messages: List[Dict[str, str]],
    *,
    max_turns: int = 24,
    max_chars: int = 1200,
) -> str:
    """Serialize prior turns for the prompt via DB-backed conversation memory helpers.

    Exact example
    -------------
    IN:  []  →  OUT: ""
    IN:  [{"role":"user","content":"What about asthma?"},
          {"role":"assistant","content":"Indicated for ages 6+."}]
    OUT includes a Previous conversation header with User/Assistant lines.

    For the first question in a new session, history is empty → "".
    """
    if not messages:
        return ""
    from app.services.conversation_memory import build_prompt_history

    return build_prompt_history(
        messages,
        recent_limit=max_turns,
        max_chars_per_msg=max_chars,
    )


def _build_chat_llm():
    """
    What: Build a LangChain chat model for the ReAct agent (HF path).
    Why:  ``create_react_agent`` needs a LangChain-compatible LLM, not raw HTTP.
    How:  Ollama → ``build_chat_ollama``; HF → ``ChatHuggingFace`` over Endpoint.
    Out:  LangChain chat model

    Connects to: ``get_agent_executor`` only (not used on Ollama direct RAG path).
    """
    if settings.uses_ollama_llm:
        from app.rag.llm.ollama_client import (
            build_chat_ollama,
            check_ollama_model,
            check_ollama_reachable,
        )

        check_ollama_reachable(settings.OLLAMA_BASE_URL)
        check_ollama_model(settings.OLLAMA_BASE_URL, settings.LLM_MODEL)
        return build_chat_ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_NEW_TOKENS,
        )

    try:
        from langchain_huggingface import HuggingFaceEndpoint
        from langchain_huggingface.chat_models import ChatHuggingFace
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace LangChain packages are not installed. "
            "Install with: pip install -r backend/requirements-dev.txt "
            "(production uses Bedrock / Ollama direct RAG)."
        ) from exc

    llm = HuggingFaceEndpoint(
        repo_id=settings.LLM_MODEL,
        max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
        timeout=300,
    )
    return ChatHuggingFace(llm=llm)


def get_agent_executor(
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
):
    """
    What: Wire PDF / math / web tools into a LangChain ReAct ``AgentExecutor``.
    Why:  On HuggingFace, the model decides when to call ``pdf_search`` etc.
    How:  Create tools → chat LLM → ReAct prompt (+ multi-doc guidance) → executor.
    In:   user/document scope, optional history
    Out:  ``(executor, pdf_tool, formatted_history)``
          ``pdf_tool.last_sources`` later becomes citation chips

    Connects to: ``generate_answer`` / ``generate_answer_stream`` HF branches.
    """

    # Initialize tools
    pdf_tool = PDFSearchTool(user_id=user_id, document_id=document_id, document_ids=document_ids, top_k=top_k)
    tools = [pdf_tool, MathTool(), WebSearchTool()]

    chat_llm = _build_chat_llm()

    # Setup Agent — prefer DB agent template + domain overlay
    try:
        resolved = _resolve_runtime_prompts(
            question="agent_setup",
            user_id=user_id,
            document_id=document_id,
            document_ids=document_ids,
            chat_history=chat_history,
            prompt_type="agent",
        )
        agent_prompt_text = resolved.task_prompt or AGENT_SYSTEM_PROMPT
        for block in (resolved.system_prompt or "").split("\n\n"):
            if block.strip().startswith("DOMAIN FOCUS"):
                agent_prompt_text = agent_prompt_text.replace(
                    "Begin!",
                    block.strip() + "\n\nBegin!",
                    1,
                )
                break
    except Exception:
        agent_prompt_text = AGENT_SYSTEM_PROMPT
    if document_ids and len(document_ids) > 1:
        agent_prompt_text = agent_prompt_text.replace(
            "Begin!",
            MULTI_DOC_COMPARISON_GUIDANCE + "\nBegin!",
            1,
        )
    prompt = PromptTemplate.from_template(agent_prompt_text)
    agent = create_react_agent(chat_llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    formatted_history = _format_chat_history(chat_history) if chat_history else ""

    return executor, pdf_tool, formatted_history

def is_greeting(question: str) -> bool:
    """True iff normalized question is in the fixed greetings set."""
    from app.services.conversation_memory import is_greeting as _is_greeting

    return _is_greeting(question)


def is_capability_question(question: str) -> bool:
    """True for product/capability questions that must not drift into document RAG."""
    from app.services.conversation_memory import (
        is_capability_question as _is_capability,
    )

    return _is_capability(question)


def is_chat_memory_question(question: str) -> bool:
    """True when the user asks to answer from chat memory / this session only."""
    from app.services.conversation_memory import (
        is_chat_memory_question as _is_memory,
    )

    return _is_memory(question)


def _maybe_quiz_intent(question: str, document_id: Optional[str], document_ids: Optional[List[str]]):
    """Return QuizIntent when the user asks for an interactive quiz."""
    from app.rag.utils.quiz import parse_quiz_intent

    return parse_quiz_intent(
        question,
        has_document=bool(document_id or document_ids),
    )


def _unique_document_ids(document_ids: Optional[List[str]]) -> List[str]:
    """Preserve order while dropping empty / duplicate IDs."""
    unique: List[str] = []
    seen = set()
    for raw in document_ids or []:
        doc_id = str(raw) if raw is not None else ""
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def _is_summarize_attached_intent(question: str) -> bool:
    """True for summarize / list / what-are-these-docs style questions."""
    q = (question or "").lower().strip()
    if not q:
        return False
    markers = (
        "summarize",
        "summarise",
        "summary",
        "overview of",
        "what are these",
        "what are the documents",
        "what documents",
        "list the documents",
        "list attached",
        "list the attached",
        "documents attached",
        "all attached",
        "all the attached",
        "tell me about these",
        "tell me about the documents",
        "what's attached",
        "whats attached",
        "which documents",
        "attached pdfs",
        "attached documents",
    )
    return any(marker in q for marker in markers)


def _load_attached_document_rows(user_id: str, document_ids: List[str]) -> List[Dict[str, Any]]:
    """Load filename + stored summary for each attached document ID."""
    if not document_ids:
        return []
    try:
        from app.database import SessionLocal
        from app.models import Document

        db = SessionLocal()
        try:
            rows = (
                db.query(Document)
                .filter(Document.user_id == user_id, Document.id.in_(document_ids))
                .all()
            )
            by_id = {str(row.id): row for row in rows}
            out: List[Dict[str, Any]] = []
            for doc_id in document_ids:
                row = by_id.get(doc_id)
                if not row:
                    continue
                out.append(
                    {
                        "id": doc_id,
                        "filename": row.original_name or "document",
                        "summary": (row.summary or "").strip(),
                    }
                )
            return out
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Failed loading attached document summaries: %s", exc)
        return []


def _build_attached_summarize_context(
    question: str,
    user_id: str,
    document_ids: List[str],
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """Build per-doc summary context for summarize/list-attached intents.

    Returns ``(context, sources)`` or ``None`` when intent does not apply.
    Falls back to 1–2 retrieved chunks per doc when ``Document.summary`` is empty.
    """
    if len(document_ids) < 1 or not _is_summarize_attached_intent(question):
        return None

    docs = _load_attached_document_rows(user_id, document_ids)
    if not docs:
        return None

    parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs, start=1):
        filename = doc["filename"]
        summary = doc["summary"]
        if not summary:
            fallback = retrieve(
                query=question,
                user_id=user_id,
                document_id=doc["id"],
                document_ids=None,
                top_k=2,
            )
            if fallback:
                summary = "\n".join(
                    (c.get("text") or "").strip() for c in fallback if c.get("text")
                ).strip()
                for chunk in fallback:
                    sources.append(
                        {
                            "text": chunk["text"][:300]
                            + ("..." if len(chunk["text"]) > 300 else ""),
                            "filename": chunk.get("filename", filename),
                            "page": chunk.get("page", "?"),
                            "score": chunk.get("score", 0),
                            "confidence": chunk.get("confidence", 0),
                            "bbox": chunk.get("bbox", ""),
                        }
                    )
            else:
                summary = "(No stored summary or retrievable chunks for this document.)"

        if not any(s.get("filename") == filename for s in sources):
            preview = summary[:300] + ("..." if len(summary) > 300 else "")
            sources.append(
                {
                    "text": preview,
                    "filename": filename,
                    # SourceChunk.page is int — use 0 for stored-summary citations.
                    "page": 0,
                    "score": 1.0,
                    "confidence": 100.0,
                    "bbox": "",
                }
            )

        parts.append(
            f"[{i}] Document: {filename}\n"
            f"Stored summary / excerpt:\n{summary}"
        )

    header = (
        "The user asked to summarize or list the attached documents. "
        "Produce a structured overview that covers EVERY document below by filename. "
        "Do not omit any document.\n\n"
    )
    return header + "\n\n".join(parts), sources


def _multi_doc_prompt_prefix(document_ids: Optional[List[str]], chunks: List[Dict[str, Any]]) -> str:
    """Light nudge so multi-attach answers cover each filename."""
    unique = _unique_document_ids(document_ids)
    if len(unique) <= 1:
        return ""
    names = []
    seen = set()
    for chunk in chunks:
        name = chunk.get("filename")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    name_line = ", ".join(names) if names else f"{len(unique)} attached documents"
    return (
        MULTI_DOC_COMPARISON_GUIDANCE.strip()
        + f"\nCover each attached document ({name_line}) when the question is broad.\n\n"
    )


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Build the evidence block for ``RAG_PROMPT_TEMPLATE`` ``{context}``.

    Exact format per chunk (code):
      f"[{i}] Source: {filename}, Page {page}\\n{text}"
      joined with "\\n\\n"

    Exact example (ILLUSTRATIVE chunk text; format is exact)
    -------------------------------------------------------
    IN chunks[0] = {
      "filename": "xolair_prescribing.pdf", "page": 9,
      "text": "The recommended dose for CSU is 150 mg or 300 mg by "
              "subcutaneous injection every 4 weeks.",
    }
    OUT starts with:
      [1] Source: xolair_prescribing.pdf, Page 9
      The recommended dose for CSU is 150 mg or 300 mg by subcutaneous
      injection every 4 weeks.

    Next: RAG_PROMPT_TEMPLATE.format(context=OUT, question=question)
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        filename = chunk.get("filename", "document")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        parts.append(f"[{i}] Source: {filename}, Page {page}\n{text}")
    return "\n\n".join(parts)


def _coerce_source_page(page: Any) -> int:
    try:
        return int(page)
    except (TypeError, ValueError):
        return 0


def _sources_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map retrieve chunks → citation payloads for the UI (text truncated to 300).

    Exact OUT keys (code): text, filename, page, score, confidence, bbox.

    Exact example
    -------------
    IN chunk text length 500 → OUT text length 303 (300 + "...")
    OUT item ILLUSTRATIVE:
      {"text": "The recommended dose for CSU is ...", "filename": "xolair_prescribing.pdf",
       "page": 9, "score": 0.0323, "confidence": 100.0, "bbox": ""}

    Stream path yields: data: {"type":"sources","data":[<items>]}\n\n  BEFORE tokens.
    """
    return [
        {
            "text": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
            "filename": chunk["filename"],
            "page": _coerce_source_page(chunk.get("page")),
            "score": float(chunk.get("score") or 0),
            "confidence": float(chunk.get("confidence") or 0),
            "bbox": chunk.get("bbox", "") or "",
        }
        for chunk in chunks
    ]


def _graph_scope(
    document_id: Optional[str], document_ids: Optional[List[str]]
) -> List[str]:
    """Documents the graph may draw evidence from for this request."""
    scope = _unique_document_ids(document_ids)
    if document_id and document_id not in scope:
        scope = [document_id] + scope
    return scope


def _collect_graph_evidence(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
):
    """Retrieve bounded, evidence-backed graph context for the question.

    Returns ``None`` when graph retrieval is disabled, empty, or fails; the
    caller then behaves exactly like the vector-only RAG pipeline.
    """
    if not getattr(settings, "GRAPH_RETRIEVAL_ENABLED", False):
        return None

    try:
        from app.database import SessionLocal
        from app.graph.retrieval import retrieve_graph_evidence
    except Exception as exc:  # optional dependency chain missing
        logger.warning("GraphRAG unavailable: %s", exc)
        return None

    db = None
    try:
        db = SessionLocal()
        evidence = retrieve_graph_evidence(
            db,
            user_id,
            question,
            document_ids=_graph_scope(document_id, document_ids) or None,
        )
    except Exception as exc:
        logger.warning("GraphRAG evidence retrieval failed: %s", exc)
        return None
    finally:
        if db is not None:
            db.close()

    if evidence is None or evidence.is_empty:
        return None
    return evidence


def _graph_payload(evidence) -> Optional[Dict[str, Any]]:
    """Compact graph metadata for the UI / chat history."""
    if evidence is None:
        return None
    names = [
        entity.get("canonical_name")
        for entity in evidence.entities
        if entity.get("canonical_name")
    ]
    return {
        "entities": names[:12],
        "relationship_count": len(evidence.relationships),
        "source_count": len(evidence.sources),
        "truncated": bool(evidence.truncated),
    }


def _merge_sources(
    sources: List[Dict[str, Any]], extra: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Append graph evidence citations without duplicating vector sources."""
    merged = list(sources or [])
    seen = {
        (
            item.get("filename"),
            item.get("page"),
            (item.get("text") or "")[:60],
        )
        for item in merged
        if isinstance(item, dict)
    }
    for item in extra or []:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        key = (
            item.get("filename"),
            item.get("page"),
            (item.get("text") or "")[:60],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _classify_turn_intent(
    question: str,
    chat_history: Optional[List[Dict[str, str]]],
    *,
    has_attach: bool,
):
    """Classify this turn so routing does not hard-gate on attachment alone."""
    from app.services.conversation_memory import classify_query_intent

    intent = classify_query_intent(
        question, chat_history, has_attach=has_attach
    )
    logger.info(
        "Query intent=%s has_attach=%s history_turns=%s q=%r",
        intent.value,
        has_attach,
        len(chat_history or []),
        (question or "")[:120],
    )
    return intent


def _should_prefer_chat_memory(
    question: str,
    chat_history: Optional[List[Dict[str, str]]],
    *,
    has_attach: bool,
) -> bool:
    """Prefer session history over corpus/document retrieval for this turn."""
    from app.services.conversation_memory import QueryIntent

    intent = _classify_turn_intent(
        question, chat_history, has_attach=has_attach
    )
    return intent is QueryIntent.CHAT_MEMORY


def _try_special_turn_answer(
    question: str,
    chat_history: Optional[List[Dict[str, str]]],
    *,
    has_attach: bool,
    llm_provider: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Handle capability and chat-memory turns before any retrieve/graph call.

    Attachment must NOT suppress capability or conversation-memory intents.
    Hybrid/document/general return None so the caller continues into RAG.
    """
    from app.services.conversation_memory import QueryIntent

    intent = _classify_turn_intent(
        question, chat_history, has_attach=has_attach
    )
    if intent is QueryIntent.CAPABILITY:
        return _capability_answer(
            question, chat_history, llm_provider=llm_provider
        )
    if intent is QueryIntent.CHAT_MEMORY:
        # Pure memory answers should not re-surface PDF citation chips unless
        # the user is explicitly asking about previously discussed document facts.
        prior = (
            _prior_sources_from_history(chat_history)
            if is_document_oriented_question_safe(question)
            else []
        )
        return _chat_memory_answer(
            question,
            chat_history,
            llm_provider=llm_provider,
            prior_sources=prior,
        )
    return None


def is_document_oriented_question_safe(question: str) -> bool:
    from app.services.conversation_memory import is_document_oriented_question

    return is_document_oriented_question(question)


def _capability_answer(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    *,
    llm_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer product/capability questions without retrieving documents.

    Prior conversation is still injected so a later follow-up about the
    document topic remains coherent after a capability digression.
    """
    provider = normalize_llm_provider(llm_provider)
    history = _format_chat_history(chat_history) if chat_history else ""
    answer = ""
    client = None
    try:
        client = get_llm_client(llm_provider=provider)
        user_content = GREETING_PROMPT.format(question=question)
        if history:
            user_content = (
                f"{history}\n\n"
                "The user is now asking about product capabilities (not document content). "
                f"{user_content}"
            )
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Document AI Analyst for Genova. Answer briefly about "
                        "product capabilities only. Do not discuss medical drugs, "
                        "doses, invoices, project names, or any document content. "
                        "Conversation history is provided only so you stay oriented; "
                        "do not digress into it unless the user asks."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            model=llm_model_for_provider(provider),
            max_tokens=256,
            temperature=settings.LLM_TEMPERATURE,
        )
        if response and getattr(response, "choices", None):
            answer = (response.choices[0].message.content or "").strip()
    except Exception:
        # Provider failures must not turn a greeting into an error —
        # fall back to the static answer below.
        answer = ""
    if not answer:
        answer = (
            "I can help you upload and analyze PDFs, answer questions with citations, "
            "summarize documents, and generate interactive quizzes. Attach a PDF to "
            "ground answers in that file."
        )
    result: Dict[str, Any] = {"answer": answer, "sources": []}
    usage = getattr(client, "last_usage", None) if client is not None else None
    if isinstance(usage, dict) and usage:
        result["usage"] = usage
    return result


def _chat_memory_answer(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    *,
    llm_provider: Optional[str] = None,
    prior_sources: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Answer from chat history only — no corpus retrieval / no document sources."""
    from app.services.conversation_memory import resolve_deterministic_memory_answer

    deterministic = resolve_deterministic_memory_answer(question, chat_history)
    if deterministic:
        return {
            "answer": deterministic,
            "sources": list(prior_sources or []),
        }

    provider = normalize_llm_provider(llm_provider)
    history = _format_chat_history(chat_history) if chat_history else ""
    user_prompt = (
        "Answer ONLY from the conversation history below. "
        "Ignore any attached PDFs or corpus for this turn — the user is asking "
        "about the chat itself or facts they already stated. "
        "Every prior User turn counts — including statements that are not phrased "
        "as questions. When asked for the first/previous question or message, "
        "quote that User turn verbatim. "
        "Do not invent document facts. If the history is insufficient, say you "
        "cannot determine the answer from this chat alone.\n\n"
        f"{history or '(No prior conversation.)'}\n\n"
        f"Current question: {question}\n\nAnswer:"
    )
    answer = ""
    client = None
    try:
        client = get_llm_client(llm_provider=provider)
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Document AI Analyst. Use only the provided chat history. "
                        "Never retrieve or invent document contents. Do not cite PDF filenames "
                        "unless they appear in the chat history."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            model=llm_model_for_provider(provider),
            max_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
        if response and getattr(response, "choices", None):
            answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        # Provider credentials (e.g. a HuggingFace user token) may be missing;
        # answer gracefully instead of failing the whole chat request.
        logger.warning("Chat-memory answer unavailable: %s", exc)
    if not answer:
        answer = MALFORMED_OUTPUT_MESSAGE
    result: Dict[str, Any] = {
        "answer": answer,
        "sources": list(prior_sources or []),
    }
    usage = getattr(client, "last_usage", None) if client is not None else None
    if isinstance(usage, dict) and usage:
        result["usage"] = usage
    return result


def _prior_sources_from_history(
    chat_history: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not chat_history:
        return []
    from app.services.conversation_memory import last_assistant_sources

    return last_assistant_sources(chat_history)


def _generate_answer_direct(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm_provider: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve-then-generate (non-streaming) for Ollama or Bedrock.

    Exact call sequence for the shared XOLAIR question
    --------------------------------------------------
    1) chunks = retrieve(query=question, user_id=..., document_id=...,
                         top_k=top_k or settings.TOP_K_RERANK)  # default 8
    2) sources = _sources_from_chunks(chunks)
    3) if not chunks → fixed insufficient-info answer, sources=[]
    4) context = _format_context(chunks)
       user_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
       (+ history prepended if any)
    5) messages = [
         {"role":"system", "content": SYSTEM_PROMPT},  # from prompts.py
         {"role":"user", "content": user_prompt},
       ]
       response = get_llm_client().chat_completion(
         messages=..., model="qwen2.5:3b", max_tokens=1024, temperature=0.3)
    6) answer = response.choices[0].message.content.strip()
       empty → MALFORMED_OUTPUT_MESSAGE

    Exact OUT: {"answer": str, "sources": list[dict], "usage"?: dict}

    ILLUSTRATIVE answer (not a live model run):
      "No. The label recommends 150 mg or 300 mg SC every 4 weeks for CSU,
       not every 2 weeks. [Source: xolair_prescribing.pdf, Page 9]"
    """
    has_attach = bool(document_id or _unique_document_ids(document_ids))
    special = _try_special_turn_answer(
        question,
        chat_history,
        has_attach=has_attach,
        llm_provider=llm_provider,
    )
    if special is not None:
        return special

    attach_ids = _unique_document_ids(document_ids)

    # No documents attached: do not search the whole user corpus (that pollutes
    # domain detection and leaks unrelated chunks). Prefer session memory, else
    # refuse with the standard insufficient-info message after logging General.
    if not has_attach:
        _resolve_runtime_prompts(
            question=question,
            user_id=user_id,
            document_id=None,
            document_ids=[],
            chat_history=chat_history,
            retrieved_context=None,
            llm_provider=llm_provider,
            prompt_type="rag",
            session_id=session_id,
        )
        if chat_history:
            return _chat_memory_answer(
                question,
                chat_history,
                llm_provider=llm_provider,
                prior_sources=_prior_sources_from_history(chat_history),
            )
        return {
            "answer": (
                "I couldn't find sufficient information in the uploaded documents "
                "to answer this question."
            ),
            "sources": [],
        }

    graph_evidence = _collect_graph_evidence(question, user_id, document_id, document_ids)
    graph_info = _graph_payload(graph_evidence)
    graph_block = f"\n\n{graph_evidence.context}" if graph_evidence else ""

    summarize_pack = (
        _build_attached_summarize_context(question, user_id, attach_ids)
        if attach_ids
        else None
    )
    if summarize_pack:
        context, sources = summarize_pack
    else:
        chunks = retrieve(
            query=question,
            user_id=user_id,
            document_id=document_id,
            document_ids=document_ids,
            top_k=top_k or settings.TOP_K_RERANK,
        )
        sources = _sources_from_chunks(chunks)
        if not chunks and not graph_evidence:
            # Fall back to session memory when corpus has nothing but history does.
            if chat_history:
                return _chat_memory_answer(
                    question,
                    chat_history,
                    llm_provider=llm_provider,
                    prior_sources=_prior_sources_from_history(chat_history),
                )
            return {
                "answer": (
                    "I couldn't find sufficient information in the uploaded documents "
                    "to answer this question."
                ),
                "sources": [],
            }
        context = (
            _multi_doc_prompt_prefix(document_ids, chunks) + _format_context(chunks)
            if chunks
            else graph_block.strip()
        )

    if graph_evidence:
        context = f"{context}{graph_block}"
        sources = _merge_sources(sources, graph_evidence.sources)

    resolved = _resolve_runtime_prompts(
        question=question,
        user_id=user_id,
        document_id=document_id,
        document_ids=document_ids,
        chat_history=chat_history,
        retrieved_context=context[:2000] if context else None,
        llm_provider=llm_provider,
        prompt_type="rag",
        session_id=session_id,
    )
    user_prompt = resolved.format_task(context=context, question=question)
    history = _format_chat_history(chat_history) if chat_history else ""
    if history:
        user_prompt = f"{history}\n\n{user_prompt}"

    provider = normalize_llm_provider(llm_provider)
    client = get_llm_client(llm_provider=provider)
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": resolved.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=llm_model_for_provider(provider),
        max_tokens=settings.LLM_MAX_NEW_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
    )
    answer = ""
    if response and getattr(response, "choices", None):
        answer = (response.choices[0].message.content or "").strip()
    if not answer:
        answer = MALFORMED_OUTPUT_MESSAGE
    result: Dict[str, Any] = {"answer": answer, "sources": sources}
    if graph_info:
        result["graph"] = graph_info
    usage = getattr(client, "last_usage", None)
    if isinstance(usage, dict) and usage:
        result["usage"] = usage
    return result


def _generate_answer_stream_direct(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm_provider: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """Streaming twin of ``_generate_answer_direct`` (SSE lines for WS/HTTP stream).

    Exact yield order
    -----------------
    1) data: {"type":"sources","data": sources}\\n\\n
    2) if no chunks: one token with insufficient-info message, then done; return
    3) same prompt build as non-stream
    4) chat_completion(..., stream=True) → for each delta:
         data: {"type":"token","data": "<piece>"}\\n\\n
       on exception: data: {"type":"error","data": "..."}\\n\\n
    5) optional data: {"type":"usage","data": {...}}\\n\\n  (Bedrock metadata)
    6) data: {"type":"done"}\\n\\n

    Exact example first events (ILLUSTRATIVE token pieces)
    ------------------------------------------------------
    data: {"type":"sources","data":[{"page":9,"filename":"xolair_prescribing.pdf",...}]}
    data: {"type":"token","data":"No"}
    data: {"type":"token","data":". The label"}
    ...
    data: {"type":"done"}
    """
    has_attach = bool(document_id or _unique_document_ids(document_ids))
    special = _try_special_turn_answer(
        question,
        chat_history,
        has_attach=has_attach,
        llm_provider=llm_provider,
    )
    if special is not None:
        yield f"data: {json.dumps({'type': 'sources', 'data': special.get('sources') or []})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'data': special.get('answer') or ''})}\n\n"
        usage = special.get("usage")
        if isinstance(usage, dict) and usage:
            yield f"data: {json.dumps({'type': 'usage', 'data': usage})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    yield f"data: {json.dumps({'type': 'status', 'data': 'retrieving'})}\n\n"
    attach_ids = _unique_document_ids(document_ids)

    if not has_attach:
        _resolve_runtime_prompts(
            question=question,
            user_id=user_id,
            document_id=None,
            document_ids=[],
            chat_history=chat_history,
            retrieved_context=None,
            llm_provider=llm_provider,
            prompt_type="rag",
            session_id=session_id,
        )
        if chat_history:
            result = _chat_memory_answer(
                question,
                chat_history,
                llm_provider=llm_provider,
                prior_sources=_prior_sources_from_history(chat_history),
            )
            yield f"data: {json.dumps({'type': 'sources', 'data': result.get('sources') or []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'data': result.get('answer') or ''})}\n\n"
            usage = result.get("usage")
            if isinstance(usage, dict) and usage:
                yield f"data: {json.dumps({'type': 'usage', 'data': usage})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        msg = (
            "I couldn't find sufficient information in the uploaded documents "
            "to answer this question."
        )
        yield f"data: {json.dumps({'type': 'sources', 'data': []})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'data': msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    graph_evidence = _collect_graph_evidence(question, user_id, document_id, document_ids)
    graph_info = _graph_payload(graph_evidence)
    graph_block = f"\n\n{graph_evidence.context}" if graph_evidence else ""
    if graph_info:
        yield f"data: {json.dumps({'type': 'graph', 'data': graph_info})}\n\n"

    summarize_pack = (
        _build_attached_summarize_context(question, user_id, attach_ids)
        if attach_ids
        else None
    )
    if summarize_pack:
        context, sources = summarize_pack
    else:
        chunks = retrieve(
            query=question,
            user_id=user_id,
            document_id=document_id,
            document_ids=document_ids,
            top_k=top_k or settings.TOP_K_RERANK,
        )
        sources = _sources_from_chunks(chunks)
        if not chunks and not graph_evidence:
            if chat_history:
                result = _chat_memory_answer(
                    question,
                    chat_history,
                    llm_provider=llm_provider,
                    prior_sources=_prior_sources_from_history(chat_history),
                )
                yield f"data: {json.dumps({'type': 'sources', 'data': result.get('sources') or []})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'data': result.get('answer') or ''})}\n\n"
                usage = result.get("usage")
                if isinstance(usage, dict) and usage:
                    yield f"data: {json.dumps({'type': 'usage', 'data': usage})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            msg = (
                "I couldn't find sufficient information in the uploaded documents "
                "to answer this question."
            )
            yield f"data: {json.dumps({'type': 'sources', 'data': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'data': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        context = (
            _multi_doc_prompt_prefix(document_ids, chunks) + _format_context(chunks)
            if chunks
            else graph_block.strip()
        )

    if graph_evidence:
        context = f"{context}{graph_block}"
        sources = _merge_sources(sources, graph_evidence.sources)

    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
    resolved = _resolve_runtime_prompts(
        question=question,
        user_id=user_id,
        document_id=document_id,
        document_ids=document_ids,
        chat_history=chat_history,
        retrieved_context=context[:2000] if context else None,
        llm_provider=llm_provider,
        prompt_type="rag",
        session_id=session_id,
    )
    user_prompt = resolved.format_task(context=context, question=question)
    history = _format_chat_history(chat_history) if chat_history else ""
    if history:
        user_prompt = f"{history}\n\n{user_prompt}"

    provider = normalize_llm_provider(llm_provider)
    try:
        yield f"data: {json.dumps({'type': 'status', 'data': 'generating'})}\n\n"
        client = get_llm_client(llm_provider=provider)
        stream = client.chat_completion(
            messages=[
                {"role": "system", "content": resolved.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=llm_model_for_provider(provider),
            max_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'type': 'token', 'data': chunk.choices[0].delta.content})}\n\n"
        usage = getattr(client, "last_usage", None)
        if isinstance(usage, dict) and usage:
            yield f"data: {json.dumps({'type': 'usage', 'data': usage})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'usage': usage})}\n\n"
            return
    except Exception as e:
        logger.error("Direct stream error (%s): %s", provider, e)
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        return

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@trace_function(
    "generate_answer",
    metadata_factory=lambda question, user_id, document_id=None, **kwargs: {
        "user_id": user_id,
        "document_id": document_id,
        "llm_model": settings.LLM_MODEL,
    },
)
def generate_answer(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm_provider: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Public non-streaming entry used by ``/chat/ask``.

    Exact branch order (code)
    -------------------------
    1) is_greeting(question) → short LLM hello, sources=[]  (no retrieve)
    2) ollama/bedrock (direct RAG) → return _generate_answer_direct(...)
         on error → ExternalServiceException(...)
    3) else HuggingFace ReAct:
         get_agent_executor → executor.invoke → parse_agent_output
         sources from pdf_tool.last_sources
         malformed → MALFORMED_OUTPUT_MESSAGE

    Exact example (default Ollama + XOLAIR question)
    ------------------------------------------------
    generate_answer(
      question=<CSU every-2-week question>,
      user_id=..., document_id=<xolair id>, chat_history=[],
    )
    → branch 2 → _generate_answer_direct → {"answer": "...", "sources": [...]}
    """
    provider = normalize_llm_provider(llm_provider)

    # ── Interactive quiz (conversation / document / summary) ─
    quiz_intent = _maybe_quiz_intent(question, document_id, document_ids)
    if quiz_intent is not None:
        from app.rag.utils.quiz import generate_quiz_response

        try:
            # Still resolve/log domain + prompt selection for observability.
            _resolve_runtime_prompts(
                question=question,
                user_id=user_id,
                document_id=document_id,
                document_ids=document_ids,
                chat_history=chat_history,
                retrieved_context=None,
                llm_provider=provider,
                prompt_type="rag",
                session_id=session_id,
            )
            return generate_quiz_response(
                question,
                user_id,
                document_id=document_id,
                document_ids=document_ids,
                chat_history=chat_history,
                top_k=top_k,
                llm_provider=provider,
                intent=quiz_intent,
            )
        except Exception as e:
            logger.error("Quiz generation error: %s", e)
            raise ExternalServiceException("Quiz", str(e)) from e

    # ── Capability / chat-memory (attachment must not suppress these) ───────
    has_attach = bool(document_id or document_ids)
    special = _try_special_turn_answer(
        question,
        chat_history,
        has_attach=has_attach,
        llm_provider=provider,
    )
    if special is not None:
        return special

    # Local Ollama / Bedrock: direct retrieve-then-generate
    if provider in {"ollama", "bedrock"} or (
        provider != "huggingface" and settings.uses_direct_rag_llm
    ):
        label = "Bedrock" if provider == "bedrock" else "Ollama"
        try:
            return _generate_answer_direct(
                question,
                user_id,
                document_id,
                document_ids,
                top_k,
                chat_history,
                llm_provider=provider,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("%s direct RAG error: %s", label, e)
            raise ExternalServiceException(label, str(e)) from e

    # ── Run HuggingFace ReAct Agent ──────────────────
    try:
        executor, pdf_tool, formatted_history = get_agent_executor(
            user_id,
            document_id=document_id,
            document_ids=document_ids,
            top_k=top_k,
            chat_history=chat_history,
        )
        result = executor.invoke({"input": question, "chat_history": formatted_history})

        raw_answer = result.get("output", "")
        try:
            answer = parse_agent_output(raw_answer)
        except OutputParserError as e:
            logger.warning(f"Rejected malformed LLM output: {e}")
            answer = MALFORMED_OUTPUT_MESSAGE

        # Retrieve sources from the PDF tool if it was used
        sources = [
            {
                "text": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
                "filename": chunk["filename"],
                "page": chunk["page"],
                "score": chunk["score"],
                "confidence": chunk.get("confidence", 0),
                "bbox": chunk.get("bbox", ""),
            }
            for chunk in getattr(pdf_tool, "last_sources", [])
        ]

        return {"answer": answer, "sources": sources}

    except (OutputParserError, ValueError) as e:
        logger.warning(f"Agent output error: {e}")
        return {"answer": MALFORMED_OUTPUT_MESSAGE, "sources": []}
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        raise ExternalServiceException("HuggingFace", str(e)) from e


@trace_function(
    "generate_answer_stream",
    metadata_factory=lambda question, user_id, document_id=None, **kwargs: {
        "user_id": user_id,
        "document_id": document_id,
        "llm_model": settings.LLM_MODEL,
    },
)
def generate_answer_stream(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm_provider: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """Public streaming entry for WebSocket ``/chat/ws`` and ``POST /chat/ask/stream``.

    Exact branch order (mirrors generate_answer)
    --------------------------------------------
    1) quiz intent → sources → tokens → quiz → done
    2) greeting → stream tokens (no sources content) → done
    3) ollama/bedrock → yield from _generate_answer_stream_direct(...)
    4) else HF ReAct executor.stream → sources once from pdf_tool → token answer → done

    Exact example (default Ollama dashboard ask)
    --------------------------------------------
    generate_answer_stream(question=<CSU question>, user_id=..., document_id=...)
      → sources event, then token events, then done
    """
    provider = normalize_llm_provider(llm_provider)

    # ── Interactive quiz ─────────────────────────────
    quiz_intent = _maybe_quiz_intent(question, document_id, document_ids)
    if quiz_intent is not None:
        from app.rag.utils.quiz import generate_quiz_stream

        try:
            _resolve_runtime_prompts(
                question=question,
                user_id=user_id,
                document_id=document_id,
                document_ids=document_ids,
                chat_history=chat_history,
                retrieved_context=None,
                llm_provider=provider,
                prompt_type="rag",
                session_id=session_id,
            )
            yield from generate_quiz_stream(
                question,
                user_id,
                document_id=document_id,
                document_ids=document_ids,
                chat_history=chat_history,
                top_k=top_k,
                llm_provider=provider,
                intent=quiz_intent,
            )
        except Exception as e:
            logger.error("Quiz stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # ── Capability / chat-memory (attachment must not suppress these) ────────
    has_attach = bool(document_id or document_ids)
    special = _try_special_turn_answer(
        question,
        chat_history,
        has_attach=has_attach,
        llm_provider=provider,
    )
    if special is not None:
        yield f"data: {json.dumps({'type': 'sources', 'data': special.get('sources') or []})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'data': special.get('answer') or ''})}\n\n"
        usage = special.get("usage")
        if isinstance(usage, dict) and usage:
            yield f"data: {json.dumps({'type': 'usage', 'data': usage})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    if provider in {"ollama", "bedrock"} or (
        provider != "huggingface" and settings.uses_direct_rag_llm
    ):
        yield from _generate_answer_stream_direct(
            question,
            user_id,
            document_id,
            document_ids,
            top_k,
            chat_history,
            llm_provider=provider,
            session_id=session_id,
        )
        return

    # ── Run HuggingFace ReAct Agent ──────────────────
    try:
        executor, pdf_tool, formatted_history = get_agent_executor(
            user_id,
            document_id=document_id,
            document_ids=document_ids,
            top_k=top_k,
            chat_history=chat_history,
        )

        sources_sent = False

        for step in executor.stream({"input": question, "chat_history": formatted_history}):
            if "actions" in step:
                continue

            elif "intermediate_steps" in step:
                if not sources_sent and getattr(pdf_tool, "last_sources", []):
                    sources = [
                        {
                            "text": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
                            "filename": chunk["filename"],
                            "page": chunk["page"],
                            "score": chunk["score"],
                            "confidence": chunk.get("confidence", 0),
                            "bbox": chunk.get("bbox", ""),
                        }
                        for chunk in pdf_tool.last_sources
                    ]
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
                    sources_sent = True

            elif "output" in step:
                full_answer = step["output"]
                try:
                    clean_answer = parse_agent_output(full_answer)
                except OutputParserError as e:
                    logger.warning(f"Rejected malformed streamed LLM output: {e}")
                    clean_answer = MALFORMED_OUTPUT_MESSAGE
                yield f"data: {json.dumps({'type': 'token', 'data': clean_answer})}\n\n"

    except Exception as e:
        logger.error(f"Agent streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# =============================================================================
# PIPELINE FLOW WITH EXACT EXAMPLE
# =============================================================================
#
# Shared question Q:
#   "Does the XOLAIR prescribing label recommend every-2-week dosing for
#    chronic spontaneous urticaria (CSU)?"
#
# A. Actual input into this file (dashboard /ask or /ask/stream)
#    question: str = Q
#    user_id: str
#    document_id: str | None   # selected PDF, e.g. xolair_prescribing.pdf
#    chat_history: list[dict]  # [] on first turn
#    top_k: Optional[int]      # usually None → agent uses TOP_K_RERANK (8)
#
# B. Ollama direct path (VERIFIED default: LLM_PROVIDER=ollama)
#    is_greeting(Q) → False
#    _generate_answer_direct / _generate_answer_stream_direct
#      retrieve(Q, user_id, document_id, top_k=8)
#      _sources_from_chunks(chunks)
#      _format_context(chunks)
#      user_prompt = RAG_PROMPT_TEMPLATE.format(context=..., question=Q)
#      messages = [SYSTEM_PROMPT, user_prompt]
#      get_llm_client().chat_completion(model="qwen2.5:3b", max_tokens=1024,
#                                       temperature=0.3, stream=...)
#    OUT non-stream: {"answer": str, "sources": list}
#    OUT stream: SSE sources → token* → done
#
# C. HuggingFace ReAct path (LLM_PROVIDER=huggingface)
#    get_agent_executor → tools [PDFSearchTool, MathTool, WebSearchTool]
#    executor.invoke/stream → parse_agent_output → sources from pdf_tool.last_sources
#
# D. Representative prompt shape (exact template fields; chunk text ILLUSTRATIVE)
#    system: SYSTEM_PROMPT (cite [Source: filename, Page X], answer only from context)
#    user: RAG_PROMPT_TEMPLATE with
#      ## Document Context
#      [1] Source: xolair_prescribing.pdf, Page 9
#      ... every 4 weeks ...
#      ## User Question
#      Q
#      ## Instructions ... ## Answer
#
# -----------------------------------------------------------------------------
# COMPLETE END-TO-END EXAMPLE (ALL SIX FILES) — ASK PATH
# -----------------------------------------------------------------------------
#
# User question → embeddings.py → vectorstore.py + bm25.py → retriever.py
#   → reranker.py (if enabled) → agent.py → LLM → final answer + sources
#
# Compact table (default config: hybrid ON, RERANKER_ENABLED=False, Ollama):
#
# Step | File/function | Input | Processing | Output | Next step
# -----|---------------|-------|------------|--------|----------
# 0 | agent.generate_answer(_stream) | Q, user_id, document_id | branch: not greeting; uses_ollama_llm | call direct helper | 1
# 1 | agent._generate_answer_* | same | top_k = top_k or TOP_K_RERANK (8) | retrieve(...) | 2
# 2 | retriever.transform_query | Q | LLM rewrite ≤3 + original; dedupe; cap 4 | list[str] search queries | 3
# 3 | embeddings.embed_query | one search_query | cache → Ollama nomic (search_query: prefix) | List[float] len 768 | 4
# 4 | vectorstore.query_chunks | embedding, user_id, document_id, top_k=8 | Chroma cosine NN; where document_id $eq; score=1-distance | list[dict] semantic hits | 5
# 5 | bm25.query_bm25 | same search_query string, filters, top_k=8 | tokenize; BM25Okapi.get_scores; raw scores | list[dict] lexical hits | 6
# 6 | retriever.rrf_merge | vector_results, bm25_results, k=60 | score += 1/(k+rank) per list | list with rrf_score → renamed score | 7
# 7 | retriever._merge_candidates | all variant hits | dedupe doc|page|text[:200] | unique candidates | 8
# 8 | reranker.get_reranker | (none) | RERANKER_ENABLED False → None | None | 9a
# 9a | retriever sort+slice | candidates | sort by score; [:TOP_K_RERANK=8]; confidence % | ≤8 chunks | 10
# 9b | (alt) Reranker.rerank | Q, candidates, top_k=8 | only if enabled; CrossEncoder pairs | ≤8 with rerank_score | 10
# 10 | agent._sources_from_chunks | chunks | truncate text 300 | sources for UI | 11
# 11 | agent._format_context | chunks | "[i] Source: file, Page N\\ntext" | context str | 12
# 12 | prompts + get_llm_client | context + Q | SYSTEM + RAG_PROMPT; chat_completion qwen2.5:3b | answer str / token stream | UI
#
# Index-time (once per upload), for completeness:
#   vectorstore.store_chunks → bm25.store_bm25_index + embed_documents + collection.add
#
# VERIFIED from config.py: nomic-embed-text/768, chroma_db_ollama_nomic, hybrid True,
#   RRF_K=60, TOP_K_RETRIEVAL=20, TOP_K_RERANK=8, RERANKER_ENABLED=False,
#   LLM qwen2.5:3b temp 0.3 max 1024.
# ILLUSTRATIVE (not live-run): exact float vectors, BM25/RRF numeric scores,
#   rewrite paraphrase strings, chunk page text, final model wording.
# =============================================================================
