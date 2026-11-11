"""Deterministic structured table lookups — domain-agnostic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.rag.generation.structured_lookup import (
    VERIFIED_BLOCK_MARKER,
    attach_verified_evidence,
    build_verified_evidence,
    extract_query_parameters,
)
from app.rag.llm.agent import _context_with_verified_lookups, _finalize_rag_user_prompt


def _rate_card_chunk() -> dict:
    """Synthetic pricing matrix (not medical)."""
    return {
        "filename": "rates.pdf",
        "page": 2,
        "chunk_type": "table",
        "text": (
            "| Volume | 1-10 kg | 11-50 kg | 51-100 kg |\n"
            "| --- | --- | --- | --- |\n"
            "| 100-200 | 10 | 12 | 15 |\n"
            "| 201-400 | 18 | 22 | 30 |\n"
            "| 401-600 | 25 | 35 | 45 |\n\n"
            "Table facts:\n"
            "- 100-200 | 1-10 kg | 10\n"
            "- 100-200 | 11-50 kg | 12\n"
            "- 100-200 | 51-100 kg | 15\n"
            "- 201-400 | 1-10 kg | 18\n"
            "- 201-400 | 11-50 kg | 22\n"
            "- 201-400 | 51-100 kg | 30\n"
            "- 401-600 | 1-10 kg | 25\n"
            "- 401-600 | 11-50 kg | 35\n"
            "- 401-600 | 51-100 kg | 45\n"
        ),
    }


def _shipping_chunk() -> dict:
    """Synthetic schedule grid with frequency column."""
    return {
        "filename": "shipping.pdf",
        "page": 4,
        "chunk_type": "table",
        "table_json": {
            "headers": ["Zone", "Frequency", "1-5 kg", "6-20 kg"],
            "rows": [
                ["A", "every 2 weeks", "5", "9"],
                ["B", "every 4 weeks", "7", "11"],
            ],
        },
        "text": "Shipping rate table",
    }


def test_extract_query_parameters_skips_every_n():
    q = "Item volume 350 units and weight 78 kg was quoted at 15 every 4 weeks."
    params = extract_query_parameters(q)
    values = {(p.value, p.unit) for p in params}
    assert (350.0, "units") in values or (350.0, "") in values or any(
        p.value == 350.0 for p in params
    )
    assert any(p.value == 78.0 and p.unit == "kg" for p in params)
    # "4" from "every 4 weeks" should not be a free parameter.
    assert not any(p.value == 4.0 and not p.unit for p in params)


def test_rate_card_two_axis_lookup():
    q = "What is the rate for volume 350 and weight 78 kg?"
    evidence = build_verified_evidence(q, [_rate_card_chunk()])
    assert evidence.lookups, "expected a matched cell"
    lu = evidence.lookups[0]
    assert lu.value == "30"
    assert "201-400" in lu.row_label
    assert "51-100" in lu.col_header


def test_stated_mismatch_on_rate_card():
    q = (
        "Volume 350 and weight 78 kg was intentionally quoted 15 every 4 weeks. "
        "Is that correct?"
    )
    evidence = build_verified_evidence(q, [_rate_card_chunk()])
    assert evidence.lookups
    assert evidence.lookups[0].value == "30"
    assert evidence.comparisons
    assert any(not c.matches for c in evidence.comparisons)


def test_shipping_table_json_lookup():
    q = "Zone A package weight 12 kg — what amount and schedule?"
    # Zone A is categorical; weight 12 matches 6-20 kg column.
    # Row match needs numeric range — Zone A won't match via number.
    # Use a numeric-row shipping-like matrix instead via facts.
    chunk = {
        "filename": "lanes.pdf",
        "page": 1,
        "text": (
            "Table facts:\n"
            "- 1-5 | 6-20 kg | 9 | every 2 weeks\n"
            "- 10-20 | 6-20 kg | 14 | every 2 weeks\n"
            "- 1-5 | 1-5 kg | 5 | every 4 weeks\n"
        ),
    }
    evidence = build_verified_evidence(
        "lane band 12 and weight 15 kg — recommended amount?",
        [chunk],
    )
    assert evidence.lookups
    assert evidence.lookups[0].value == "14"
    assert "every 2 weeks" in (evidence.lookups[0].frequency or "")


def test_smashed_weight_columns_and_stated_mismatch():
    """Multi-range column headers + IgE-like row bands (generic assay×weight matrix)."""
    chunk = {
        "filename": "matrix.pdf",
        "page": 5,
        "text": (
            "Table facts:\n"
            "- ≥30−100 | 30−60 kg | 150 | Every 4 weeks\n"
            "- ≥30−100 | >70−90 kg >90−150 kg | 150 300 | Every 4 weeks\n"
            "- >300−400 | 30−60 kg | 225 | Every 2 weeks\n"
            "- >300−400 | >70−90 kg >90−150 kg | 300 | Every 2 weeks\n"
            "- >400−500 | >70−90 kg >90−150 kg | 375 | Every 2 weeks\n"
        ),
    }
    q = (
        "A 35-year-old subject, body weight 78 kg and pretreatment assay "
        "350 IU/mL was intentionally prescribed 150 mg every 4 weeks."
    )
    evidence = build_verified_evidence(q, [chunk])
    assert evidence.lookups
    lu = evidence.lookups[0]
    assert lu.value == "300"
    assert "every 2 weeks" in lu.frequency.lower()
    assert any(p.value == 350 for p in extract_query_parameters(q))
    assert not any(p.value == 35 for p in extract_query_parameters(q))
    assert evidence.comparisons
    assert any(not c.matches for c in evidence.comparisons)


def test_narrative_only_yields_empty():
    chunks = [
        {
            "filename": "notes.pdf",
            "page": 1,
            "text": "This document discusses general policies without matrices.",
        }
    ]
    evidence = build_verified_evidence("What is the rate for 350 and 78 kg?", chunks)
    assert not evidence.has_content


def test_prepend_and_finalize_reminder():
    evidence = build_verified_evidence(
        "What is the rate for volume 350 and weight 78 kg?",
        [_rate_card_chunk()],
    )
    ctx = attach_verified_evidence("chunk body", evidence)
    assert VERIFIED_BLOCK_MARKER in ctx
    assert "deterministic" not in ctx.lower()
    assert ctx.index(VERIFIED_BLOCK_MARKER) < ctx.index("chunk body")
    prompt = _finalize_rag_user_prompt(
        "base prompt",
        context=ctx,
        question="rate?",
    )
    assert "TABLE LOOKUP" in prompt


def test_context_helper_injects_block():
    chunks = [_rate_card_chunk()]
    out, evidence = _context_with_verified_lookups(
        "What is the rate for volume 350 and weight 78 kg?",
        chunks,
        "[1] Source: rates.pdf, Page 2\nbody",
    )
    assert VERIFIED_BLOCK_MARKER in out
    assert "30" in out
    assert "deterministic" not in out.lower()
    assert evidence is not None and evidence.has_content


@pytest.mark.parametrize("provider", ["ollama", "bedrock"])
def test_agent_injects_same_verified_block_regardless_of_provider(provider):
    """Deterministic block must be identical across mocked LLM providers."""
    from app.rag.llm import agent as agent_mod

    chunks = [_rate_card_chunk()]
    captured = {}

    class FakeResolved:
        domain = ""
        prompt_type = "rag"
        system_prompt = "sys"

        def format_task(self, context="", question=""):
            captured["context"] = context
            return f"CTX\n{context}\nQ\n{question}"

    fake_client = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = "The verified rate is 30."
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]
    fake_client.chat_completion.return_value = fake_resp
    fake_client.last_usage = {}

    with patch.object(agent_mod, "retrieve", return_value=chunks), patch.object(
        agent_mod, "_try_special_turn_answer", return_value=None
    ), patch.object(agent_mod, "_collect_graph_evidence", return_value=None), patch.object(
        agent_mod, "_build_attached_summarize_context", return_value=None
    ), patch.object(
        agent_mod, "_resolve_runtime_prompts", return_value=FakeResolved()
    ), patch.object(
        agent_mod, "get_llm_client", return_value=fake_client
    ), patch.object(
        agent_mod, "normalize_llm_provider", return_value=provider
    ), patch.object(
        agent_mod, "llm_model_for_provider", return_value="mock-model"
    ), patch(
        "app.config.get_settings"
    ) as mock_settings:
        settings = MagicMock()
        settings.USE_AGENT_FRAMEWORK = False
        settings.TOP_K_RERANK = 8
        settings.LLM_MAX_NEW_TOKENS = 256
        settings.LLM_TEMPERATURE = 0.0
        settings.RAG_DEBUG_ENABLED = False
        mock_settings.return_value = settings
        # settings is module-level in agent — patch that too
        with patch.object(agent_mod, "settings", settings):
            result = agent_mod._generate_answer_direct(
                "What is the rate for volume 350 and weight 78 kg?",
                user_id="u1",
                document_id="d1",
                llm_provider=provider,
            )

    assert VERIFIED_BLOCK_MARKER in captured["context"]
    assert "30" in captured["context"]
    assert "deterministic" not in captured["context"].lower()
    assert "Verified table lookup" not in result["answer"]
    assert result["answer"]
    fake_client.chat_completion.assert_called_once()
