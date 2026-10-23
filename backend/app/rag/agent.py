"""
Agentic RAG — intelligent routing using ReAct (Reasoning and Acting).
Intelligently chooses between PDF search, Web Search, and Math tools.

LLM backends:
- ollama (default): local Ollama chat model — no HF_TOKEN required
- huggingface: HuggingFace Inference API — requires HF_TOKEN
"""
import logging
import json
from typing import List, Dict, Any, Optional, Generator

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from app.config import get_settings
from app.rag.retriever import retrieve
from app.rag.graph_retriever import get_entity_context
from app.rag.prompts import (
    AGENT_SYSTEM_PROMPT,
    MULTI_DOC_COMPARISON_GUIDANCE,
    RAG_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from app.exceptions import ExternalServiceException
from app.rag.security import MALFORMED_OUTPUT_MESSAGE, OutputParserError, parse_agent_output
from app.rag.tools import PDFSearchTool, MathTool, WebSearchTool
from app.rag.tracing import trace_function
from app.rag.keywords import extract_keywords

logger = logging.getLogger(__name__)
settings = get_settings()


def persist_document_keywords(document, chunks, db) -> None:
    """Extract TF-IDF keywords after indexing and persist on the document row."""
    raw_texts = [c["text"] for c in chunks]
    kws = extract_keywords(raw_texts, top_n=10)
    document.keywords = json.dumps(kws)
    db.add(document)


def get_llm_client(hf_token: Optional[str] = None):
    """Return a chat client for the configured LLM provider.

    Ollama: ``OllamaChatClient`` (InferenceClient-compatible ``chat_completion``).
    HuggingFace: ``huggingface_hub.InferenceClient`` (requires token).
    """
    if settings.uses_ollama_llm:
        from app.rag.ollama_client import (
            OllamaChatClient,
            check_ollama_model,
            check_ollama_reachable,
        )

        check_ollama_reachable(settings.OLLAMA_BASE_URL)
        check_ollama_model(settings.OLLAMA_BASE_URL, settings.LLM_MODEL)
        return OllamaChatClient(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL,
        )

    from huggingface_hub import InferenceClient

    token = hf_token or settings.HF_TOKEN
    if not token:
        raise ValueError(
            "Hugging Face API token is missing. Set HF_TOKEN, or switch to "
            "local Ollama with LLM_PROVIDER=ollama (the default)."
        )
    return InferenceClient(token=token)


def _format_chat_history(messages: List[Dict[str, str]]) -> str:
    if not messages:
        return ""
    lines = ["Previous conversation:"]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_chat_llm(hf_token: Optional[str] = None):
    """Build the LangChain chat model for the ReAct agent."""
    if settings.uses_ollama_llm:
        from app.rag.ollama_client import (
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

    from langchain_huggingface import HuggingFaceEndpoint
    from langchain_huggingface.chat_models import ChatHuggingFace

    token = hf_token or settings.HF_TOKEN
    if not token:
        raise ValueError(
            "Hugging Face API token is missing. Set HF_TOKEN, or switch to "
            "local Ollama with LLM_PROVIDER=ollama (the default)."
        )

    llm = HuggingFaceEndpoint(
        repo_id=settings.LLM_MODEL,
        huggingfacehub_api_token=token,
        max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
        timeout=300,
    )
    return ChatHuggingFace(llm=llm)


def get_agent_executor(
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    hf_token: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
):
    """Initialize the LangChain ReAct agent executor."""

    # Initialize tools
    pdf_tool = PDFSearchTool(user_id=user_id, document_id=document_id, document_ids=document_ids, top_k=top_k)
    tools = [pdf_tool, MathTool(), WebSearchTool()]

    chat_llm = _build_chat_llm(hf_token)

    # Setup Agent
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
    """Detect if the question is a casual greeting rather than a document query."""
    greetings = {
        "hi", "hello", "hey", "how are you", "what's up", "whats up",
        "good morning", "good evening", "good afternoon", "thanks", "thank you",
        "bye", "goodbye", "help", "what can you do", "who are you",
    }
    return question.lower().strip().rstrip("!?.") in greetings


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        filename = chunk.get("filename", "document")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        parts.append(f"[{i}] Source: {filename}, Page {page}\n{text}")
    return "\n\n".join(parts)


def _sources_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "text": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
            "filename": chunk["filename"],
            "page": chunk["page"],
            "score": chunk["score"],
            "confidence": chunk.get("confidence", 0),
            "bbox": chunk.get("bbox", ""),
        }
        for chunk in chunks
    ]


def _generate_answer_direct(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Retrieve-then-generate path for local Ollama models.

    Small chat models often fail ReAct tool-calling; this path preserves
    hybrid retrieval, citations, and grounding without requiring tool use.
    """
    chunks = retrieve(
        query=question,
        user_id=user_id,
        document_id=document_id,
        document_ids=document_ids,
        top_k=top_k or settings.TOP_K_RERANK,
    )
    sources = _sources_from_chunks(chunks)
    if not chunks:
        return {
            "answer": (
                "I couldn't find sufficient information in the uploaded documents "
                "to answer this question."
            ),
            "sources": [],
        }

    context = _format_context(chunks)
    user_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    history = _format_chat_history(chat_history) if chat_history else ""
    if history:
        user_prompt = f"{history}\n\n{user_prompt}"

    client = get_llm_client()
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=settings.LLM_MODEL,
        max_tokens=settings.LLM_MAX_NEW_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
    )
    answer = ""
    if response and getattr(response, "choices", None):
        answer = (response.choices[0].message.content or "").strip()
    if not answer:
        answer = MALFORMED_OUTPUT_MESSAGE
    return {"answer": answer, "sources": sources}


def _generate_answer_stream_direct(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Generator[str, None, None]:
    """Streaming variant of the direct Ollama RAG path."""
    chunks = retrieve(
        query=question,
        user_id=user_id,
        document_id=document_id,
        document_ids=document_ids,
        top_k=top_k or settings.TOP_K_RERANK,
    )
    sources = _sources_from_chunks(chunks)
    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

    if not chunks:
        msg = (
            "I couldn't find sufficient information in the uploaded documents "
            "to answer this question."
        )
        yield f"data: {json.dumps({'type': 'token', 'data': msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    context = _format_context(chunks)
    user_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    history = _format_chat_history(chat_history) if chat_history else ""
    if history:
        user_prompt = f"{history}\n\n{user_prompt}"

    try:
        client = get_llm_client()
        stream = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'type': 'token', 'data': chunk.choices[0].delta.content})}\n\n"
    except Exception as e:
        logger.error(f"Ollama direct stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

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
    hf_token: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Agentic generation: retrieve via tools → reason → generate answer.
    """
    # ── Handle greetings ─────────────────────────────
    if is_greeting(question):
        client = get_llm_client(hf_token)
        try:
            messages = [
                {"role": "system", "content": "You are Document AI Analyst, a friendly AI assistant."},
                {"role": "user", "content": question},
            ]
            response = client.chat_completion(
                messages=messages,
                model=settings.LLM_MODEL,
                max_tokens=256,
            )
            answer = response.choices[0].message.content.strip() if response.choices else "Hello! How can I help you today?"
        except Exception:
            answer = "Hello! I'm Document AI Analyst. How can I help you with your documents?"
        return {"answer": answer, "sources": []}

    # Local Ollama: direct retrieve-then-generate (more reliable than ReAct on small models)
    if settings.uses_ollama_llm:
        try:
            return _generate_answer_direct(
                question, user_id, document_id, document_ids, top_k, chat_history
            )
        except Exception as e:
            logger.error(f"Ollama direct RAG error: {e}")
            raise ExternalServiceException("Ollama", str(e)) from e

    # ── Run HuggingFace ReAct Agent ──────────────────
    try:
        executor, pdf_tool, formatted_history = get_agent_executor(user_id, document_id, document_ids, hf_token, top_k, chat_history)
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
    hf_token: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Generator[str, None, None]:
    """
    Streaming Agentic pipeline.
    """
    # ── Handle greetings ─────────────────────────────
    if is_greeting(question):
        yield f"data: {json.dumps({'type': 'sources', 'data': []})}\n\n"
        client = get_llm_client(hf_token)
        try:
            stream = client.chat_completion(
                messages=[{"role": "user", "content": question}],
                model=settings.LLM_MODEL,
                max_tokens=256,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.choices[0].delta.content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    if settings.uses_ollama_llm:
        yield from _generate_answer_stream_direct(
            question, user_id, document_id, document_ids, top_k, chat_history
        )
        return

    # ── Run HuggingFace ReAct Agent ──────────────────
    try:
        executor, pdf_tool, formatted_history = get_agent_executor(user_id, document_id, document_ids, hf_token, top_k, chat_history)

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
