import re

from app.rag.retrieval import retriever


def test_transform_query_includes_original_and_dedupes(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "_generate_query_variants",
        lambda _query: [
            "How do taxes work?",
            "how do taxes work?",
            "How does healthcare work?",
            "healthcare overview",
        ],
    )

    # Default: LLM rewrite OFF — only original (+ deterministic structured variants).
    queries = retriever.transform_query("How do taxes and healthcare work?")
    assert queries[0] == "How do taxes and healthcare work?"
    assert "How do taxes work?" not in queries

    # Explicit allow_llm_rewrite enables paraphrases.
    queries_llm = retriever.transform_query(
        "How do taxes and healthcare work?",
        allow_llm_rewrite=True,
    )
    assert queries_llm == [
        "How do taxes and healthcare work?",
        "How do taxes work?",
        "How does healthcare work?",
        "healthcare overview",
    ]


def test_retrieve_fans_out_transformed_queries_and_merges_duplicates(monkeypatch):
    from unittest.mock import MagicMock
    searched_queries = []

    mock_db = MagicMock()
    mock_db.__enter__.return_value = mock_db
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value.all.return_value = [("doc-1",)]
    monkeypatch.setattr("app.database.SessionLocal", lambda: mock_db)

    monkeypatch.setattr(
        retriever,
        "transform_query",
        lambda _query, **_kw: ["taxes", "healthcare"],
    )
    monkeypatch.setattr(retriever, "embed_query", lambda query: f"embedding:{query}")
    monkeypatch.setattr(retriever, "get_reranker", lambda: None)

    def fake_query_chunks(query_embedding, user_id, document_id=None, document_ids=None, top_k=10):
        searched_queries.append(query_embedding)
        if query_embedding == "embedding:taxes":
            return [
                {
                    "id": "shared",
                    "text": "Shared chunk",
                    "filename": "policy.pdf",
                    "page": 1,
                    "score": 0.2,
                },
                {
                    "id": "taxes",
                    "text": "Tax chunk",
                    "filename": "policy.pdf",
                    "page": 2,
                    "score": 0.7,
                },
            ]

        return [
            {
                "id": "shared",
                "text": "Shared chunk",
                "filename": "policy.pdf",
                "page": 1,
                "score": 0.9,
            },
            {
                "id": "healthcare",
                "text": "Healthcare chunk",
                "filename": "policy.pdf",
                "page": 3,
                "score": 0.8,
            },
        ]

    monkeypatch.setattr(retriever, "query_chunks", fake_query_chunks)

    chunks = retriever.retrieve("How do taxes and healthcare work?", user_id="user-1")

    assert searched_queries == ["embedding:taxes", "embedding:healthcare"]
    assert [chunk["id"] for chunk in chunks] == ["shared", "taxes", "healthcare"]
    assert chunks[0]["score"] > 0  # RRF score, not raw similarity
    assert chunks[0]["id"] == "shared"  # highest RRF score — appears in both query results
    assert chunks[0]["confidence"] == 100.0


def test_retrieve_includes_active_but_excludes_deleted(monkeypatch):
    from unittest.mock import MagicMock
    
    mock_db = MagicMock()
    mock_db.__enter__.return_value = mock_db
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value.all.return_value = [("doc-active",)]
    monkeypatch.setattr("app.database.SessionLocal", lambda: mock_db)
    
    monkeypatch.setattr(
        retriever, "transform_query", lambda _query, **_kw: ["test"]
    )
    monkeypatch.setattr(retriever, "embed_query", lambda query: f"embedding:{query}")
    monkeypatch.setattr(retriever, "get_reranker", lambda: None)
    
    captured_ids = []
    def fake_query_chunks(query_embedding, user_id, document_id=None, document_ids=None, top_k=10):
        captured_ids.append(document_ids)
        return [{"id": "chunk-1", "text": "sample text", "filename": "policy.pdf", "page": 1, "score": 0.5}]
        
    monkeypatch.setattr(retriever, "query_chunks", fake_query_chunks)
    
    retriever.retrieve("hello", user_id="user-1")
    
    assert captured_ids == [None]  # retrieve() passes document_id=None, not document_ids


def test_recover_caption_from_chunk_text():
    cap = retriever._recover_chunk_caption(
        {
            "text": (
                "Dosing Information\n"
                "Table 1. Subcutaneous Doses for Patients 12 Years of Age "
                "and Older with Asthma\n"
                "| IgE | Dose |"
            )
        }
    )
    assert cap.startswith("Table 1.")
    assert "Asthma" in cap


def test_boost_prefers_caption_over_sticky_subsection():
    """Wrong sticky subsection must not beat the table's own caption."""
    q = (
        "A 16-year-old patient with asthma has pretreatment IgE of 450 IU/mL "
        "and weighs 85 kg. What dose and frequency are recommended?"
    )
    candidates = [
        {
            "id": "t3",
            "chunk_type": "table",
            "text": "Table 3. Doses for Adult Patients with Condition B\n| 450 | 375 |",
            "caption": "Table 3. Doses for Adult Patients with Condition B",
            # Spurious sticky heading from a sibling block:
            "subsection": "Recommended Dosage for Condition C",
            "section": "Dosing Information",
            "score": 0.05,
        },
        {
            "id": "t1",
            "chunk_type": "table",
            "text": (
                "Table 1. Subcutaneous Doses for Patients 12 Years of Age "
                "and Older with Asthma\n| 450 | 375 |"
            ),
            "caption": (
                "Table 1. Subcutaneous Doses for Patients 12 Years of Age "
                "and Older with Asthma"
            ),
            "section": "Dosing Information",
            "subsection": "",
            "score": 0.05,
        },
    ]
    boosted = retriever._boost_table_candidates(q, candidates, modality="table")
    by_id = {c["id"]: c for c in boosted}
    # Caption "asthma" on T1 must beat sticky "recommended" on T3.
    assert by_id["t1"]["score"] > by_id["t3"]["score"]
    boosted.sort(key=lambda x: x.get("score", 0), reverse=True)
    assert boosted[0]["id"] == "t1"

def test_caption_number_coverage_prefers_band_matching_title():
    """Query number covered by caption band boosts that table (P8-style)."""
    q = (
        "An 8-year-old child with asthma weighs 35 kg and has pretreatment "
        "IgE of 350 IU/mL. What is the labeled dose and frequency?"
    )
    candidates = [
        {
            "id": "t1",
            "chunk_type": "table",
            "caption": (
                "Table 1. Doses for Patients 12 Years of Age and Older with Asthma"
            ),
            "text": "Table 1. Doses for Patients 12 Years of Age and Older with Asthma",
            "score": 0.05,
        },
        {
            "id": "t2",
            "chunk_type": "table",
            "caption": (
                "Table 2. Doses for Pediatric Patients with Asthma Between "
                "the Ages of 6 to <12 Years"
            ),
            "text": (
                "Table 2. Doses for Pediatric Patients with Asthma Between "
                "the Ages of 6 to <12 Years"
            ),
            "score": 0.05,
        },
    ]
    # Routing often returns BOTH for these asks — boost must still run.
    for modality in ("table", "both"):
        boosted = retriever._boost_table_candidates(
            q, candidates, modality=modality
        )
        by_id = {c["id"]: c for c in boosted}
        assert by_id["t2"]["score"] > by_id["t1"]["score"], modality


def test_lookup_query_masks_unanchored_age_and_prescribed_amount():
    """A6-style: keep IgE/weight nums; mask age / prescribed / cadence."""
    q = (
        "A 35-year-old asthma patient weighing 78 kg with pretreatment IgE of "
        "350 IU/mL was intentionally prescribed XOLAIR 150 mg every 4 weeks."
    )
    chunks = [
        {
            "chunk_type": "table",
            "table_json": (
                '{"headers":["Pretreatment Serum IgE (IU/mL)","Dosing Freq.",'
                '"Body Weight / >70-90 kg / Dose (mg)",'
                '"Body Weight / >90-150 kg / Dose (mg)"],'
                '"rows":[[">300-400","Every 2 weeks","300","Insufficient Data"],'
                '["≥30-100","Every 4 weeks","150","300"]]}'
            ),
        }
    ]
    out = retriever._lookup_query_for_hydrate(q, chunks)
    # Anchored
    assert "78" in out
    assert "350" in out
    # Unanchored → masked
    assert "35-year" not in out.replace("N-year", "")
    assert re.search(r"\b35\b", out) is None
    assert re.search(r"\b150\b", out) is None
    assert re.search(r"\b4\b", out) is None
    assert "N" in out


def test_coalesce_caption_sibling_scores_lifts_structured_table():
    """Prose Table-1 hit must not leave the structured T1 row under-boosted."""
    candidates = [
        {
            "id": "prose",
            "chunk_type": "text",
            "text": "Table 1. Doses for Patients with Asthma\nSee dosing below.",
            "score": 0.033,
        },
        {
            "id": "t1",
            "chunk_type": "table",
            "table_id": "t1",
            "caption": "Table 1. Doses for Patients with Asthma",
            "table_json": '{"caption":"Table 1. Doses for Patients with Asthma","headers":["a"],"rows":[["1"]]}',
            "score": 0.031,
        },
        {
            "id": "t3",
            "chunk_type": "table",
            "table_id": "t3",
            "caption": "Table 3. Doses for Adult Patients with Condition B",
            "table_json": '{"caption":"Table 3. Doses for Adult Patients with Condition B","headers":["a"],"rows":[["1"]]}',
            "score": 0.032,
        },
    ]
    q = (
        "Adult allergic asthma, serum IgE 350 IU/mL, weight 78 kg. "
        "What is the recommended subcutaneous dose?"
    )
    boosted = retriever._boost_table_candidates(q, candidates, modality="both")
    by_id = {c["id"]: c for c in boosted}
    assert by_id["t1"]["score"] > by_id["t3"]["score"]
    assert by_id["t1"].get("_caption_score_coalesced") is True


def test_lookup_query_keeps_thousands_and_caption_number_bigrams():
    """C15-style: ``1,200`` stays one number; caption ``Table 3`` id is kept."""
    q = (
        "An adult patient has pretreatment IgE of 1,200 IU/mL and weighs "
        "140 kg. Does Table 3 provide a recommended dose?"
    )
    chunks = [
        {
            "chunk_type": "table",
            "caption": "Table 3. Doses for Adult Patients with Condition B",
            "table_json": (
                '{"caption":"Table 3. Doses for Adult Patients with Condition B",'
                '"headers":["Pretreatment Serum IgE (IU/mL)","Dosing Freq.",'
                '"Body Weight / >90-125 kg / Dose (mg)",'
                '"Body Weight / >125-150 kg / Dose (mg)"],'
                '"rows":[[">1100-1200","Every 2 Weeks","300","Insufficient Data"],'
                '["≥30-100","Every 4 weeks","150","150"]]}'
            ),
        }
    ]
    out = retriever._lookup_query_for_hydrate(q, chunks)
    assert "1200" in out
    assert "1,200" not in out
    assert "140" in out
    assert "Table 3" in out
    assert "Table N" not in out


def test_number_spans_joins_thousands_separator():
    spans = retriever._number_spans("IgE of 1,200 IU/mL and 140 kg")
    vals = [v for v, *_ in spans]
    assert 1200.0 in vals
    assert 140.0 in vals
    assert 1.0 not in vals
    assert 200.0 not in vals


def test_boost_backfills_caption_from_text_when_metadata_missing():
    q = "patient with asthma. What dose is recommended?"
    candidates = [
        {
            "id": "legacy",
            "chunk_type": "table",
            "text": "Table 1. Doses for Patients with Asthma\n| a | b |",
            # No caption metadata (old index)
            "section": "Dosing Information",
            "subsection": "Recommended Dosage for Unrelated Condition",
            "score": 0.04,
        },
        {
            "id": "noise",
            "chunk_type": "table",
            "text": "Table 9. Frequency of Asthma Exacerbations",
            "section": "Clinical Studies",
            "subsection": "Asthma Trial 1",
            "score": 0.04,
        },
    ]
    boosted = retriever._boost_table_candidates(q, candidates, modality="table")
    legacy = next(c for c in boosted if c["id"] == "legacy")
    assert legacy.get("caption", "").startswith("Table 1.")
    boosted.sort(key=lambda x: x.get("score", 0), reverse=True)
    assert boosted[0]["id"] == "legacy"

