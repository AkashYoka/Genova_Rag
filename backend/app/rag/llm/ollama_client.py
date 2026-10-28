"""
Ollama HTTP clients for embeddings and chat.

Talks to a local Ollama server (default http://localhost:11434) using httpx —
same approach as a typical local RAG stack. No HuggingFace token required.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Generator, Iterable, List, Optional, Sequence

import httpx

logger = logging.getLogger(__name__)

_QWEN3_QUERY_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)


def _model_family(model: str) -> str:
    name = (model or "").strip().lower()
    if name.startswith("qwen3-embedding"):
        return "qwen3"
    if name.startswith("nomic-embed"):
        return "nomic"
    return "generic"


def check_ollama_reachable(base_url: str, timeout: float = 5.0) -> None:
    """Raise RuntimeError with a clear message if Ollama is not reachable."""
    url = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{url}/api/version")
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {url}. "
            "Start Ollama (e.g. the Ollama Windows app) and confirm "
            f"{url}/api/tags responds."
        ) from exc


def check_ollama_model(base_url: str, model: str, timeout: float = 10.0) -> None:
    """Raise RuntimeError if the named model is not installed in Ollama."""
    url = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{url}/api/tags")
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Cannot list Ollama models at {url}/api/tags. Is Ollama running?"
        ) from exc

    # Accept exact name or tagless match (nomic-embed-text == nomic-embed-text:latest)
    wanted = model.strip()
    base = wanted.split(":")[0]
    if wanted in names or any(n == wanted or n.startswith(base + ":") for n in names):
        return
    raise RuntimeError(
        f"Ollama model {wanted!r} is not installed. "
        f"Run: ollama pull {wanted}\n"
        f"Installed models: {sorted(names) or '(none)'}"
    )


class OllamaEmbeddings:
    """LangChain-compatible embeddings via Ollama /api/embed."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        batch_size: Optional[int] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._family = _model_family(model)
        if batch_size is not None:
            self.batch_size = batch_size
        elif self._family == "nomic":
            self.batch_size = 16
        else:
            self.batch_size = 8
        self._timeout = httpx.Timeout(180.0, connect=10.0)

    def _prefix(self, text: str, *, for_query: bool) -> str:
        if self._family == "nomic":
            return ("search_query: " if for_query else "search_document: ") + text
        if self._family == "qwen3":
            return (_QWEN3_QUERY_INSTRUCT + text) if for_query else text
        return text

    def _embed(self, texts: Sequence[str], *, for_query: bool) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = [
                    self._prefix(t, for_query=for_query)
                    for t in texts[start : start + self.batch_size]
                ]
                try:
                    resp = client.post(
                        "/api/embed",
                        json={"model": self.model, "input": batch, "keep_alive": "5m"},
                    )
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    raise RuntimeError(
                        f"Ollama embeddings failed ({self.base_url}/api/embed, "
                        f"model={self.model}). Is Ollama running and is the model pulled? "
                        f"Try: ollama pull {self.model}"
                    ) from exc
                data = resp.json()
                embeddings = data.get("embeddings")
                if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                    raise RuntimeError(f"Unexpected Ollama embedding response: {data!r}")
                vectors.extend(embeddings)
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, for_query=False)

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed([text], for_query=True)
        return vectors[0] if vectors else []


class OllamaChatClient:
    """Minimal InferenceClient-compatible wrapper around Ollama /api/chat."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = httpx.Timeout(300.0, connect=10.0)

    def chat_completion(
        self,
        *,
        messages: List[dict],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stream: bool = False,
        **_kwargs: Any,
    ):
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": "10m",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if stream:
            return self._stream(payload)
        try:
            with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
                resp = client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama chat failed ({self.base_url}/api/chat, model={payload['model']}). "
                f"Is Ollama running and is the model pulled? Try: ollama pull {payload['model']}"
            ) from exc

        content = (data.get("message") or {}).get("content") or ""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def _stream(self, payload: dict) -> Generator[Any, None, None]:
        try:
            with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
                with client.stream("POST", "/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    import json as _json

                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data = _json.loads(line)
                        piece = (data.get("message") or {}).get("content") or ""
                        if piece:
                            yield SimpleNamespace(
                                choices=[
                                    SimpleNamespace(
                                        delta=SimpleNamespace(content=piece)
                                    )
                                ]
                            )
                        if data.get("done"):
                            break
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama streaming chat failed ({self.base_url}/api/chat, "
                f"model={payload['model']}). Is Ollama running?"
            ) from exc


def build_chat_ollama(base_url: str, model: str, temperature: float, max_tokens: int):
    """Return a LangChain chat model backed by Ollama."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama  # type: ignore

    return ChatOllama(
        base_url=base_url.rstrip("/"),
        model=model,
        temperature=temperature,
        num_predict=max_tokens,
    )
