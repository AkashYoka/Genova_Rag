"""Caption-sibling injection via token overlap only (no domain parsers)."""
from __future__ import annotations

from app.rag.table_rag.hydrate import (
    _caption_rank_score,
    _inject_caption_matched_siblings,
    hydrate_retrieved_tables,
)


def test_caption_rank_prefers_higher_token_overlap():
    q = (
        "patient with asthma. State the labeled pediatric dose."
    )
    t1 = (
        "Table 1. Subcutaneous XOLAIR Doses Every 2 or 4 Weeks* for Patients "
        "12 Years of Age and Older with Asthma"
    )
    t2 = (
        "Table 2. Subcutaneous XOLAIR Doses Every 2 or 4 Weeks* for Pediatric "
        "Patients with Asthma Who Begin XOLAIR Between the Ages of 6 to <12 Year"
    )
    t4 = (
        "Table 4. Subcutaneous XOLAIR Doses Every 2 or 4 Weeks* for Adult and "
        "Pediatric Patients 1 Year of Age and Older with IgE-Mediated Food Allergy"
    )
    s1 = _caption_rank_score(q, t1)
    s2 = _caption_rank_score(q, t2)
    s4 = _caption_rank_score(q, t4)
    # pediatric + asthma in both query and T2 caption → highest overlap
    assert s2[1] >= s1[1]
    assert s2[1] >= s4[1]


def test_inject_sibling_when_caption_overlap_beats_retrieved(monkeypatch):
    q = "patient with asthma. State the labeled pediatric dose."
    items = [
        {
            "document_id": "doc-1",
            "table_id": "t1",
            "table": {"caption": "Table 1. Patients 12 Years of Age and Older with Asthma"},
            "caption": "Table 1. Patients 12 Years of Age and Older with Asthma",
        }
    ]
    siblings = [
        {
            "document_id": "doc-1",
            "table_id": "t1",
            "caption": "Table 1. Patients 12 Years of Age and Older with Asthma",
            "table": {"caption": "Table 1. Patients 12 Years of Age and Older with Asthma"},
        },
        {
            "document_id": "doc-1",
            "table_id": "t2",
            "caption": (
                "Table 2. Pediatric Patients with Asthma Between the Ages of "
                "6 to <12 Years"
            ),
            "summary_text": "pediatric asthma dosing",
            "page_start": 6,
            "table": {
                "caption": (
                    "Table 2. Pediatric Patients with Asthma Between the Ages of "
                    "6 to <12 Years"
                ),
                "page": 6,
                "rows": [["a"]],
            },
        },
    ]
    monkeypatch.setattr(
        "app.rag.table_rag.hydrate.list_document_tables",
        lambda **kwargs: siblings,
    )
    out = _inject_caption_matched_siblings(
        items, user_id="u1", query=q, max_inject=2
    )
    ids = {str(x.get("table_id")) for x in out}
    assert "t1" in ids
    assert "t2" in ids
    injected = [x for x in out if x.get("_sibling_caption_inject")]
    assert injected and injected[0]["table_id"] == "t2"


def test_hydrate_orders_higher_overlap_caption_first(monkeypatch):
    q = "patient with asthma. Labeled pediatric dose?"
    chunks = [
        {
            "document_id": "doc-1",
            "table_id": "t1",
            "chunk_type": "table",
            "page": 5,
            "text": "dosing",
        }
    ]
    loaded = {
        "doc-1:t1": {
            "table_id": "t1",
            "caption": "Table 1. Patients 12 Years of Age and Older with Asthma",
            "table": {
                "caption": "Table 1. Patients 12 Years of Age and Older with Asthma",
                "headers": ["IgE"],
                "rows": [["300"]],
                "page": 5,
            },
            "pages": [5],
        }
    }
    siblings = [
        {
            "document_id": "doc-1",
            "table_id": "t1",
            "caption": "Table 1. Patients 12 Years of Age and Older with Asthma",
            "table": loaded["doc-1:t1"]["table"],
            "page_start": 5,
        },
        {
            "document_id": "doc-1",
            "table_id": "t2",
            "caption": (
                "Table 2. Pediatric Patients with Asthma Between the Ages of "
                "6 to <12 Years"
            ),
            "summary_text": "ped",
            "page_start": 6,
            "table": {
                "caption": (
                    "Table 2. Pediatric Patients with Asthma Between the Ages of "
                    "6 to <12 Years"
                ),
                "headers": ["IgE", "30-40"],
                "rows": [[">300-400", "300"]],
                "page": 6,
            },
        },
    ]
    monkeypatch.setattr(
        "app.rag.table_rag.hydrate.load_tables_by_ids",
        lambda **kwargs: loaded,
    )
    monkeypatch.setattr(
        "app.rag.table_rag.hydrate.list_document_tables",
        lambda **kwargs: siblings,
    )
    out = hydrate_retrieved_tables(chunks, user_id="u1", query=q)
    assert out
    assert str(out[0].get("table_id")) == "t2"
    block = out[0].get("table_prompt_block") or ""
    assert "TABLE STUB" not in block
