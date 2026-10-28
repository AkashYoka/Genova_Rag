"""Focused tests for Ollama provider defaults and clients."""
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings, get_settings
from app.rag.llm.ollama_client import (
    OllamaChatClient,
    OllamaEmbeddings,
    check_ollama_model,
    check_ollama_reachable,
)


def test_ollama_settings_defaults_without_env(monkeypatch):
    """Local defaults: Ollama + nomic-embed-text + qwen2.5:3b, no .env required."""
    for var in (
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "LLM_MODEL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSION",
        "CHROMA_PERSIST_DIR",
        "OLLAMA_BASE_URL",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)

    get_settings.cache_clear()
    try:
        s = Settings()
        assert s.LLM_PROVIDER == "ollama"
        assert s.EMBEDDING_PROVIDER == "ollama"
        assert s.LLM_MODEL == "qwen2.5:3b"
        assert s.EMBEDDING_MODEL == "nomic-embed-text"
        assert s.EMBEDDING_DIMENSION == 768
        assert "chroma_db_ollama_nomic" in s.CHROMA_PERSIST_DIR
        assert s.OLLAMA_BASE_URL == "http://localhost:11434"
        assert s.HF_TOKEN == ""
        assert s.uses_ollama_llm is True
        assert s.uses_ollama_embeddings is True
    finally:
        get_settings.cache_clear()


def test_huggingface_provider_still_requires_token_semantics(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_TOKEN", "")
    get_settings.cache_clear()
    try:
        from app.rag.llm import agent as agent_mod

        # Reload settings on the module
        agent_mod.settings = Settings()
        with pytest.raises(ValueError, match="Hugging Face API token"):
            agent_mod.get_llm_client()
    finally:
        get_settings.cache_clear()
        from app.rag.llm import agent as agent_mod
        agent_mod.settings = get_settings()


def test_check_ollama_reachable_error_message():
    with patch("app.rag.llm.ollama_client.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client.get.side_effect = Exception("connection refused")
        # Make HTTPError path: raise httpx.HTTPError subclass
        import httpx

        mock_client.get.side_effect = httpx.ConnectError("connection refused")
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
            check_ollama_reachable("http://localhost:11434")


def test_check_ollama_model_missing():
    import httpx

    with patch("app.rag.llm.ollama_client.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"models": [{"name": "other:latest"}]}
        mock_client.get.return_value = resp
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="not installed"):
            check_ollama_model("http://localhost:11434", "nomic-embed-text")


def test_ollama_embeddings_prefixes_nomic():
    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    assert emb._prefix("hello", for_query=True).startswith("search_query:")
    assert emb._prefix("hello", for_query=False).startswith("search_document:")


def test_ollama_embeddings_calls_api():
    import httpx

    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    fake_vector = [0.1] * 768

    with patch("app.rag.llm.ollama_client.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"embeddings": [fake_vector]}
        mock_client.post.return_value = resp
        mock_client_cls.return_value = mock_client

        docs = emb.embed_documents(["alpha"])
        assert docs == [fake_vector]
        query = emb.embed_query("beta")
        assert query == fake_vector

        # Both calls used /api/embed
        assert mock_client.post.call_count == 2
        args, kwargs = mock_client.post.call_args_list[0]
        assert args[0] == "/api/embed"
        assert kwargs["json"]["model"] == "nomic-embed-text"


def test_ollama_chat_client_non_stream():
    client = OllamaChatClient(base_url="http://localhost:11434", model="qwen2.5:3b")

    with patch("app.rag.llm.ollama_client.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"message": {"content": "hello from ollama"}}
        mock_client.post.return_value = resp
        mock_client_cls.return_value = mock_client

        result = client.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
        assert result.choices[0].message.content == "hello from ollama"
        args, kwargs = mock_client.post.call_args
        assert args[0] == "/api/chat"
        assert kwargs["json"]["model"] == "qwen2.5:3b"


def test_ollama_direct_rag_uses_retrieve_and_sources(monkeypatch):
    """Ollama path should retrieve first (not ReAct) and return citations."""
    from app.rag.llm import agent as agent_mod

    monkeypatch.setattr(agent_mod.settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(agent_mod.settings, "LLM_MODEL", "qwen2.5:3b")

    fake_chunks = [
        {
            "text": "The capital of Averia is Silverport.",
            "filename": "test.pdf",
            "page": 1,
            "score": 0.91,
            "confidence": 91,
        }
    ]

    class FakeClient:
        def chat_completion(self, **kwargs):
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content="Silverport [Source: test.pdf, Page 1]"))]
            )

    monkeypatch.setattr(agent_mod, "retrieve", lambda **kwargs: fake_chunks)
    monkeypatch.setattr(agent_mod, "get_llm_client", lambda hf_token=None, llm_provider=None: FakeClient())
    monkeypatch.setattr(agent_mod, "get_agent_executor", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ReAct should not run")))

    result = agent_mod.generate_answer("What is the capital of Averia?", "user1", "doc1")
    assert "Silverport" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["filename"] == "test.pdf"
