"""
LLM-backed, domain-agnostic entity and relationship extraction.

The prompt never names a document domain and never proposes a closed list of
entity or relationship types; it asks the model to describe whatever the
document actually contains and to quote supporting text. Everything the model
returns is validated by :mod:`app.graph.schemas` before it reaches storage.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.config import get_settings
from app.graph.schemas import (
    ExtractionResult,
    GraphExtractionError,
    parse_extraction_payload,
)

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_CHUNK_CHARS = 1400
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge-graph extraction engine.

Read the supplied document excerpts and return ONLY a JSON object with this shape:
{
  "entities": [
    {
      "key": "local-reference-id",
      "name": "canonical entity name",
      "type": "ENTITY_TYPE",
      "description": "one sentence describing this entity using only the excerpt",
      "aliases": ["other names used for it in the excerpt"],
      "confidence": 0.0,
      "chunk_id": "id of the excerpt that supports it",
      "evidence": "short verbatim quote that mentions it"
    }
  ],
  "relationships": [
    {
      "source": "entity key",
      "target": "entity key",
      "type": "relationship_type",
      "description": "one sentence describing the relationship",
      "confidence": 0.0,
      "chunk_id": "id of the excerpt that supports it",
      "evidence": "short verbatim quote that states the relationship"
    }
  ]
}

Rules:
1. Extract people, organisations, products, technologies, systems, components,
   data entities, concepts, processes, events, documents, locations, and any
   other meaningful concept. Choose types that fit the document; they are free
   text and will be normalised later. Do not force a fixed vocabulary.
2. Only extract what the excerpts support. Never use outside knowledge.
3. Every relationship must be stated or directly implied by the quoted evidence
   text. Never create a relationship merely because two entities appear in the
   same excerpt.
4. Relationships are directed. "A uses B" is not "B uses A". Do not emit the
   reverse direction unless the text states it.
5. Use the same entity key for repeated mentions of the same thing so
   relationships can reference it.
6. ``confidence`` is a number between 0 and 1 reflecting extraction certainty.
7. If nothing is extractable, return {"entities": [], "relationships": []}.
8. Output JSON only — no prose, no markdown fences.
"""


def _chunk_record(chunk: Dict[str, Any], document_id: str) -> Dict[str, Any]:
    chunk_index = chunk.get("chunk_index", 0)
    return {
        "chunk_id": chunk.get("chunk_id") or f"{document_id}_{chunk_index}",
        "document_id": str(document_id),
        "page": chunk.get("page"),
        "section": chunk.get("section") or chunk.get("heading") or "",
        "text": (chunk.get("text") or "").strip(),
    }


def sample_chunks(
    chunks: Sequence[Dict[str, Any]], document_id: str, max_chunks: int
) -> List[Dict[str, Any]]:
    """Keep extraction cost bounded while covering the whole document.

    Chunks are taken at an even stride rather than only from the front, so a
    300-page PDF contributes evidence from beginning to end.
    """
    records = [_chunk_record(chunk, document_id) for chunk in chunks]
    records = [record for record in records if record["text"]]
    if max_chunks <= 0 or len(records) <= max_chunks:
        return records

    step = len(records) / float(max_chunks)
    selected = [records[min(int(index * step), len(records) - 1)] for index in range(max_chunks)]
    # Preserve document order for readable prompts.
    selected.sort(key=lambda record: str(record["page"] or ""))
    return selected


def build_prompt_batch(records: Sequence[Dict[str, Any]]) -> str:
    """Render one batch of chunk records as the user message."""
    parts = [
        "Extract entities and evidence-backed relationships from the excerpts below.",
        "Every excerpt is labelled with its chunk_id; cite that id in your output.",
        "",
    ]
    for record in records:
        text = record["text"][:MAX_CHUNK_CHARS]
        parts.append(
            f'<excerpt chunk_id="{record["chunk_id"]}" page="{record["page"]}">\n'
            f"{text}\n"
            "</excerpt>"
        )
        parts.append("")
    return "\n".join(parts).strip()


def _strip_fences(raw: str) -> str:
    text = _FENCE.sub("", (raw or "").strip())
    return text.strip()


def _first_json_object(text: str) -> str:
    """Return the outermost JSON object found in a model response."""
    start = text.find("{")
    if start == -1:
        raise GraphExtractionError("Model response contained no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise GraphExtractionError("Model response contained truncated JSON")


def parse_model_output(raw: str) -> Dict[str, Any]:
    """Parse a model response into a dict, tolerating fences and prose."""
    text = _strip_fences(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = json.loads(_first_json_object(text))
    if not isinstance(parsed, dict):
        raise GraphExtractionError("Model response was not a JSON object")
    return parsed


def _chat(
    client: Any,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
) -> str:
    response = client.chat_completion(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    if response is None:
        raise GraphExtractionError("LLM returned no response")
    choices = getattr(response, "choices", None)
    if not choices:
        raise GraphExtractionError("LLM response had no choices")
    content = getattr(choices[0].message, "content", None)
    if not content:
        raise GraphExtractionError("LLM response was empty")
    return content


def extract_from_chunks(
    chunks: Sequence[Dict[str, Any]],
    document_id: str,
    *,
    client: Optional[Any] = None,
    llm_provider: Optional[str] = None,
    max_chunks: Optional[int] = None,
    batch_size: Optional[int] = None,
    min_confidence: Optional[float] = None,
    completer: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> ExtractionResult:
    """Extract entities + relationships from document chunks.

    ``completer`` lets callers (and tests) inject the LLM call. When omitted the
    application's configured provider is used.
    """
    max_chunks = settings.GRAPH_EXTRACTION_MAX_CHUNKS if max_chunks is None else max_chunks
    batch_size = settings.GRAPH_EXTRACTION_BATCH_CHUNKS if batch_size is None else batch_size
    min_confidence = settings.GRAPH_MIN_CONFIDENCE if min_confidence is None else min_confidence

    records = sample_chunks(chunks, document_id, max_chunks)
    result = ExtractionResult(chunks_sent=len(records))
    if not records:
        return result

    chunk_lookup = {record["chunk_id"]: record for record in records}

    if completer is None:
        from app.rag.llm.agent import (
            get_llm_client,
            llm_model_for_provider,
            normalize_llm_provider,
        )

        provider = normalize_llm_provider(llm_provider)
        model = settings.GRAPH_EXTRACTION_MODEL or llm_model_for_provider(provider)
        client = client or get_llm_client(llm_provider=provider)

        def completer(messages: List[Dict[str, str]]) -> str:  # type: ignore[misc]
            return _chat(client, messages, model, max_tokens=2048)

    batch_size = max(1, batch_size)
    last_error: Optional[str] = None

    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        result.batches_attempted += 1
        prompt = build_prompt_batch(batch)
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = completer(messages)
            parsed = parse_extraction_payload(
                parse_model_output(raw),
                chunk_lookup,
                min_confidence=min_confidence,
                default_chunk_id=batch[0]["chunk_id"],
            )
        except Exception as exc:  # malformed output, timeout, provider error
            result.failed_batches += 1
            last_error = str(exc)[:300]
            logger.warning(
                "Graph extraction batch %s failed for document %s: %s",
                result.batches_attempted,
                document_id,
                exc,
            )
            continue

        result.entities.extend(parsed.entities)
        result.relationships.extend(parsed.relationships)
        result.rejected_entities += parsed.rejected_entities
        result.rejected_relationships += parsed.rejected_relationships

    if result.batches_attempted and result.failed_batches == result.batches_attempted:
        raise GraphExtractionError(
            f"All {result.batches_attempted} graph extraction batches failed"
            + (f": {last_error}" if last_error else "")
        )

    # Relationship endpoints are resolved against stored entities by the
    # service layer; anything that fails resolution is dropped there.
    return result


__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "build_prompt_batch",
    "extract_from_chunks",
    "parse_model_output",
    "sample_chunks",
]
