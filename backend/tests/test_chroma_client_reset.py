"""Regression: stale Chroma handles after concurrent ingest must reopen."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _import_real_vectorstore():
    fake = sys.modules.get("app.rag.vectordb.vectorstore")
    if fake is not None:
        del sys.modules["app.rag.vectordb.vectorstore"]
    import app.rag.vectordb.vectorstore as vs

    return vs, fake


def test_chroma_retry_resets_client_on_error_finding_id(monkeypatch):
    vs, fake = _import_real_vectorstore()
    try:
        resets = []

        def _fake_reset(*, reason: str = "") -> None:
            resets.append(reason)

        monkeypatch.setattr(vs, "reset_chroma_client", _fake_reset)
        monkeypatch.setattr(vs, "_CHROMA_MAX_ATTEMPTS", 3)
        monkeypatch.setattr(vs, "_CHROMA_RETRY_BASE_SEC", 0.0)

        calls = {"n": 0}

        def _fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception(
                    "Error executing plan: Internal error: Error finding id"
                )
            return "ok"

        class _NullCtx:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(vs, "_chroma_exclusive", lambda: _NullCtx())

        assert vs._chroma_retry("collection.query", _fn) == "ok"
        assert len(resets) == 2
        assert calls["n"] == 3
    finally:
        if fake is not None:
            sys.modules["app.rag.vectordb.vectorstore"] = fake


def test_reset_chroma_client_clears_handle(monkeypatch):
    vs, fake = _import_real_vectorstore()
    try:
        vs._chroma_client = MagicMock(name="stale")
        vs.reset_chroma_client(reason="unit-test")
        assert vs._chroma_client is None
    finally:
        if fake is not None:
            sys.modules["app.rag.vectordb.vectorstore"] = fake
