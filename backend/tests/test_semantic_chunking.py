"""Tests for LLM-assisted hierarchical semantic chunking.

These cover block extraction, strict validation/repair, deterministic fallback,
chunk reconstruction, token limits, table splitting, persistence, and the
hierarchy/reindex APIs. No live Bedrock call is made.
"""
import json
from types import SimpleNamespace

import pytest

from app.models import Document, DocumentChunk, DocumentSection
from app.rag.chunking.semantic.analyzer import (
    SemanticAnalyzer,
    deterministic_sections,
    validate_and_repair,
)
from app.rag.chunking.semantic.blocks import extract_source_blocks
from app.rag.chunking.semantic.generator import generate_chunks
from app.rag.chunking.semantic.hierarchy import build_hierarchy, to_tree
from app.rag.chunking.semantic.hierarchy_store import persist_hierarchy
from app.rag.chunking.semantic.types import (
    AnalysisResult,
    SemanticChunkList,
    SectionSpec,
    SourceBlock,
    estimate_tokens,
)
from app.rag.ingestion.pdf_layout.types import DocumentElement


# ── Helpers ─────────────────────────────────────────────────────────────────


def _elements() -> list[DocumentElement]:
    return [
        DocumentElement(
            element_type="heading",
            text="1 Introduction",
            page_start=1,
            page_end=1,
            level=1,
            section="1 Introduction",
        ),
        DocumentElement(
            element_type="paragraph",
            text="Introduction paragraph. " * 12,
            page_start=1,
            page_end=1,
            section="1 Introduction",
        ),
        DocumentElement(
            element_type="heading",
            text="1.1 Background",
            page_start=1,
            page_end=2,
            level=2,
            section="1 Introduction",
            subsection="1.1 Background",
        ),
        DocumentElement(
            element_type="paragraph",
            text="Background paragraph. " * 16,
            page_start=2,
            page_end=2,
            section="1 Introduction",
            subsection="1.1 Background",
        ),
        DocumentElement(
            element_type="table",
            text="| Drug | Dose |\n| --- | --- |\n| A | 10 |",
            page_start=2,
            page_end=2,
            section="1 Introduction",
            subsection="1.1 Background",
            content_type="table",
            metadata={
                "table": {
                    "table_id": "table-1",
                    "headers": ["Drug", "Dose"],
                    "rows": [["A", "10"]],
                    "caption": "Dosing",
                }
            },
        ),
    ]


def _blocks(count: int = 4) -> list[SourceBlock]:
    return [
        SourceBlock(
            block_id=f"b{index:05d}",
            order=index - 1,
            element_type="paragraph",
            text=f"Paragraph {index}. " + ("content " * 20),
            page_start=1,
            page_end=1,
        )
        for index in range(1, count + 1)
    ]


def _settings(**overrides):
    values = {
        "CHUNK_STRATEGY": "semantic_llm",
        "SEMANTIC_CHUNKING_ENABLED": True,
        "SEMANTIC_CHUNKING_PROVIDER": "bedrock",
        "SEMANTIC_CHUNKING_MODEL_ID": "",
        "BEDROCK_MODEL_ID": "fake-nova",
        "BEDROCK_REGION": "us-east-1",
        "SEMANTIC_CHUNKING_MAX_INPUT_TOKENS": 6000,
        "SEMANTIC_CHUNKING_MAX_OUTPUT_TOKENS": 1024,
        "SEMANTIC_CHUNKING_MAX_BLOCKS_PER_CALL": 120,
        "SEMANTIC_CHUNKING_BLOCK_PREVIEW_CHARS": 200,
        "SEMANTIC_CHUNKING_RETRIES": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeClient:
    model = "fake-nova"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _valid_payload(block_ids):
    return json.dumps(
        {
            "sections": [
                {
                    "id": "s1",
                    "title": "Root",
                    "level": 1,
                    "parent_id": None,
                    "block_ids": block_ids[:2],
                    "summary": "Root summary",
                },
                {
                    "id": "s2",
                    "title": "Child",
                    "level": 2,
                    "parent_id": "s1",
                    "block_ids": block_ids[2:],
                    "summary": "Child summary",
                },
            ]
        }
    )


# ── Block extraction ────────────────────────────────────────────────────────


def test_extract_source_blocks_stable_ids_pages_and_tables():
    blocks = extract_source_blocks(_elements())

    assert [block.block_id for block in blocks] == [
        "b00001",
        "b00002",
        "b00003",
        "b00004",
        "b00005",
    ]
    assert blocks[0].is_heading and blocks[0].level == 1
    assert blocks[2].is_heading and blocks[2].level == 2
    assert blocks[4].is_table
    assert blocks[4].table["table_id"] == "table-1"
    assert blocks[4].page_start == 2
    # Deterministic ids: extracting the same elements twice is identical.
    assert [b.block_id for b in extract_source_blocks(_elements())] == [
        b.block_id for b in blocks
    ]


# ── Deterministic fallback + validation ─────────────────────────────────────


def test_deterministic_sections_follow_heading_levels():
    blocks = extract_source_blocks(_elements())
    specs = deterministic_sections(blocks, document_name="doc.pdf")

    assert [spec.title for spec in specs] == ["1 Introduction", "1.1 Background"]
    assert [spec.level for spec in specs] == [1, 2]
    assert specs[0].block_ids == ["b00001", "b00002"]
    assert specs[1].block_ids == ["b00003", "b00004", "b00005"]


def test_validate_and_repair_dedupes_and_recovers_missing_blocks():
    blocks = _blocks(5)
    specs = [
        SectionSpec(title="A", level=1, block_ids=["b00001", "b00002", "b00002", "b99999"]),
        SectionSpec(title="B", level=2, block_ids=["b00003"]),
    ]

    repaired, stats = validate_and_repair(specs, blocks)

    assert stats["duplicate_count"] == 1
    assert stats["unknown_block_ids"] == 1
    assert stats["missing_block_count"] == 2
    assert stats["source_coverage"] == 1.0
    assigned = [bid for spec in repaired for bid in spec.block_ids]
    assert sorted(assigned) == [block.block_id for block in blocks]
    assert len(assigned) == len(set(assigned))


def test_validate_and_repair_creates_section_when_llm_returns_nothing():
    blocks = _blocks(3)
    repaired, stats = validate_and_repair([], blocks)

    assert len(repaired) == 1
    assert repaired[0].block_ids == ["b00001", "b00002", "b00003"]
    assert stats["source_coverage"] == 1.0


# ── Hierarchy + chunk reconstruction ────────────────────────────────────────


def test_build_hierarchy_resolves_parents_pages_and_tables():
    blocks = extract_source_blocks(_elements())
    specs = deterministic_sections(blocks)
    nodes = build_hierarchy(specs, blocks, document_id="doc-1")

    roots = to_tree(nodes)
    assert len(roots) == 1
    root = roots[0]
    assert root.title == "1 Introduction"
    assert root.page_start == 1 and root.page_end == 1
    assert len(root.children) == 1
    child = root.children[0]
    assert child.parent_section_id == root.section_id
    assert child.path == ["1 Introduction", "1.1 Background"]
    assert child.page_start == 1 and child.page_end == 2
    assert len(child.tables) == 1
    assert child.tables[0]["table_id"] == "table-1"
    assert child.section_id.startswith("sec_")


def test_generate_chunks_preserves_source_content_and_metadata():
    blocks = extract_source_blocks(_elements())
    nodes = build_hierarchy(
        deterministic_sections(blocks), blocks, document_id="doc-1"
    )

    chunks = generate_chunks(
        nodes,
        blocks,
        document_id="doc-1",
        document_name="doc.pdf",
        index_version="semantic-llm-v1",
        max_tokens=1800,
    )

    assert chunks
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0]["id"] == "doc-1_0"
    for chunk in chunks:
        assert chunk["section_id"]
        assert chunk["source_block_ids"]
        assert chunk["index_version"] == "semantic-llm-v1"
        assert estimate_tokens(chunk["text"]) <= 1800
    combined = "\n".join(chunk["text"] for chunk in chunks)
    assert "1.1 Background" in combined
    assert "| Drug | Dose |" in combined
    assert "Background paragraph." in combined


def test_generate_chunks_respects_token_limit_and_splits_large_tables():
    rows = [[f"row-{index}", str(index)] for index in range(80)]
    table = {
        "table_id": "big-table",
        "headers": ["Item", "Value"],
        "rows": rows,
        "caption": "Large table",
    }
    table_text = (
        "| Item | Value |\n| --- | --- |\n"
        + "\n".join(f"| {row[0]} | {row[1]} |" for row in rows)
    )
    block = SourceBlock(
        block_id="b00001",
        order=0,
        element_type="table",
        text=table_text,
        page_start=1,
        page_end=3,
        content_type="table",
        table=table,
    )
    nodes = build_hierarchy(
        [SectionSpec(title="Table section", level=1, block_ids=["b00001"])],
        [block],
        document_id="doc-1",
    )

    chunks = generate_chunks(
        nodes, [block], document_id="doc-1", max_tokens=300
    )

    assert len(chunks) > 1
    assert all("| Item | Value |" in chunk["text"] for chunk in chunks)
    assert [chunk.get("table_part") for chunk in chunks] == list(
        range(1, len(chunks) + 1)
    )
    assert all(estimate_tokens(chunk["text"]) <= 300 for chunk in chunks)


# ── LLM analyzer behavior ───────────────────────────────────────────────────


def test_analyzer_accepts_fenced_json_and_maps_parents():
    blocks = _blocks(4)
    client = _FakeClient([f"```json\n{_valid_payload([b.block_id for b in blocks])}\n```"])
    analyzer = SemanticAnalyzer(settings=_settings(), client=client)

    result = analyzer.analyze(blocks, document_name="doc.pdf", page_count=1)

    assert result.used_llm is True
    assert result.used_fallback is False
    assert [spec.title for spec in result.sections] == ["Root", "Child"]
    assert result.sections[1].parent_ref.endswith("s1")
    assert result.llm_calls == 1


def test_analyzer_retries_then_succeeds():
    blocks = _blocks(4)
    client = _FakeClient(
        ["not json", _valid_payload([b.block_id for b in blocks])]
    )
    analyzer = SemanticAnalyzer(
        settings=_settings(SEMANTIC_CHUNKING_RETRIES=1), client=client
    )

    result = analyzer.analyze(blocks, document_name="doc.pdf", page_count=1)

    assert result.used_fallback is False
    assert result.llm_calls == 2
    assert result.retries == 1


def test_analyzer_falls_back_to_deterministic_sections_on_malformed_output():
    blocks = extract_source_blocks(_elements())
    client = _FakeClient(["<html>not json</html>"])
    analyzer = SemanticAnalyzer(
        settings=_settings(SEMANTIC_CHUNKING_RETRIES=0), client=client
    )

    result = analyzer.analyze(blocks, document_name="doc.pdf", page_count=2)

    assert result.used_fallback is True
    assert [spec.title for spec in result.sections] == [
        "1 Introduction",
        "1.1 Background",
    ]
    assert result.errors


def test_analyzer_windows_long_documents():
    blocks = _blocks(10)
    client = _FakeClient(
        [
            _valid_payload([b.block_id for b in blocks[:5]]),
            _valid_payload([b.block_id for b in blocks[5:]]),
        ]
    )
    analyzer = SemanticAnalyzer(
        settings=_settings(SEMANTIC_CHUNKING_MAX_BLOCKS_PER_CALL=5),
        client=client,
    )

    result = analyzer.analyze(blocks, document_name="doc.pdf", page_count=1)

    assert result.windows >= 2
    assert result.llm_calls >= 2


# ── Service integration ─────────────────────────────────────────────────────


def test_should_use_semantic_chunking_only_for_real_pdfs(tmp_path):
    from app.rag.chunking.semantic.service import should_use_semantic_chunking

    pdf = tmp_path / "real.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")

    assert should_use_semantic_chunking(str(pdf)) is True
    assert should_use_semantic_chunking(str(text)) is False
    assert should_use_semantic_chunking("report.pdf") is False


def test_should_use_semantic_chunking_respects_strategy_override(tmp_path):
    from app.rag.chunking.semantic.service import should_use_semantic_chunking

    pdf = tmp_path / "real.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    assert should_use_semantic_chunking(str(pdf), strategy="semantic_llm") is True
    assert should_use_semantic_chunking(str(pdf), strategy="deterministic") is True
    assert should_use_semantic_chunking(str(pdf), strategy="legacy") is False


def test_analyzer_deterministic_strategy_skips_llm():
    blocks = extract_source_blocks(_elements())
    client = _FakeClient(["should never be used"])
    analyzer = SemanticAnalyzer(settings=_settings(), client=client)

    result = analyzer.analyze(
        blocks,
        document_name="doc.pdf",
        page_count=2,
        strategy="deterministic",
    )

    assert result.used_llm is False
    assert result.llm_calls == 0
    assert client.calls == []
    assert [spec.title for spec in result.sections] == [
        "1 Introduction",
        "1.1 Background",
    ]


def test_semantic_chunk_document_returns_list_compatible_result(monkeypatch):
    from app.rag.chunking.semantic import service

    elements = _elements()
    parsed = SimpleNamespace(
        elements=elements,
        chunks=[{"text": "legacy", "page": 1, "chunk_index": 0}],
        page_count=2,
    )

    def fake_parse_pdf(*args, **kwargs):
        return parsed

    def fake_analyze(blocks, **kwargs):
        specs = deterministic_sections(blocks, document_name="doc.pdf")
        return AnalysisResult(sections=specs, used_llm=True, windows=1, llm_calls=1)

    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.parser.parse_pdf", fake_parse_pdf
    )
    monkeypatch.setattr(service, "analyze_blocks", fake_analyze)

    result = service.semantic_chunk_document(
        "doc.pdf", document_id="doc-1", document_name="doc.pdf"
    )

    assert isinstance(result, list)
    assert isinstance(result, SemanticChunkList)
    assert result.hierarchy
    assert result.report["source_coverage"] == 1.0
    assert result.report["chunk_count"] == len(result)
    assert result.index_version
    assert result[0]["document_id"] == "doc-1"


# ── Persistence + APIs ──────────────────────────────────────────────────────


def _seed_hierarchy(db_session, document, user):
    blocks = extract_source_blocks(_elements())
    nodes = build_hierarchy(
        deterministic_sections(blocks), blocks, document_id=str(document.id)
    )
    chunks = generate_chunks(
        nodes,
        blocks,
        document_id=str(document.id),
        document_name=document.original_name,
        index_version="semantic-llm-v1",
        max_tokens=1800,
    )
    for chunk in chunks:
        chunk["token_count"] = estimate_tokens(chunk["text"])
    persist_hierarchy(
        document_id=str(document.id),
        user_id=str(user.id),
        sections=nodes,
        chunks=chunks,
        index_version="semantic-llm-v1",
        db=db_session,
    )
    return nodes, chunks


def test_persist_hierarchy_replaces_old_rows(db_session, ready_document, user):
    nodes, chunks = _seed_hierarchy(db_session, ready_document, user)
    assert (
        db_session.query(DocumentSection)
        .filter_by(document_id=str(ready_document.id))
        .count()
        == len(nodes)
    )
    assert (
        db_session.query(DocumentChunk)
        .filter_by(document_id=str(ready_document.id))
        .count()
        == len(chunks)
    )

    _seed_hierarchy(db_session, ready_document, user)

    assert (
        db_session.query(DocumentSection)
        .filter_by(document_id=str(ready_document.id))
        .count()
        == len(nodes)
    )
    assert (
        db_session.query(DocumentChunk)
        .filter_by(document_id=str(ready_document.id))
        .count()
        == len(chunks)
    )


def test_hierarchy_api_returns_expandable_tree(
    client, db_session, ready_document, user, auth_headers
):
    nodes, chunks = _seed_hierarchy(db_session, ready_document, user)

    response = client.get(
        f"/api/v1/documents/{ready_document.id}/hierarchy",
        headers=auth_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == str(ready_document.id)
    assert payload["section_count"] == len(nodes)
    assert len(payload["sections"]) == 1
    root = payload["sections"][0]
    assert root["title"] == "1 Introduction"
    assert len(root["children"]) == 1
    assert root["children"][0]["title"] == "1.1 Background"
    assert root["children"][0]["chunk_count"] > 0


def test_section_chunks_tables_and_related_apis(
    client, db_session, ready_document, user, auth_headers
):
    nodes, _ = _seed_hierarchy(db_session, ready_document, user)
    root = to_tree(nodes)[0]
    child = root.children[0]

    chunks_response = client.get(
        f"/api/v1/documents/{ready_document.id}/sections/{child.section_id}/chunks",
        headers=auth_headers,
    )
    assert chunks_response.status_code == 200
    chunks_payload = chunks_response.json()
    assert chunks_payload["chunk_count"] > 0
    assert all(
        chunk["section_id"] == child.section_id for chunk in chunks_payload["chunks"]
    )

    tables_response = client.get(
        f"/api/v1/documents/{ready_document.id}/sections/{child.section_id}/tables",
        headers=auth_headers,
    )
    assert tables_response.status_code == 200
    assert tables_response.json()["tables"][0]["table_id"] == "table-1"

    related_response = client.get(
        f"/api/v1/documents/{ready_document.id}/sections/{child.section_id}/related",
        headers=auth_headers,
    )
    assert related_response.status_code == 200
    related = related_response.json()
    assert related["parent"]["section_id"] == root.section_id
    assert related["previous"] is None
    assert related["next"] is None


def test_index_status_and_reindex_api(
    client, db_session, ready_document, user, auth_headers, monkeypatch
):
    _seed_hierarchy(db_session, ready_document, user)
    ready_document.index_version = "semantic-llm-v1"
    ready_document.chunk_strategy = "semantic_llm"
    ready_document.chunking_report = json.dumps({"source_coverage": 1.0})
    db_session.commit()

    status_response = client.get(
        f"/api/v1/documents/{ready_document.id}/index-status",
        headers=auth_headers,
    )
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["index_version"] == "semantic-llm-v1"
    assert status_payload["report"]["source_coverage"] == 1.0
    assert status_payload["section_count"] > 0

    from app.tasks import process_document

    monkeypatch.setattr(
        process_document,
        "delay",
        lambda **kwargs: SimpleNamespace(id="task-123"),
    )

    reindex_response = client.post(
        f"/api/v1/documents/{ready_document.id}/reindex",
        json={"strategy": "deterministic"},
        headers=auth_headers,
    )
    assert reindex_response.status_code == 200
    reindex_payload = reindex_response.json()
    assert reindex_payload["task_id"] == "task-123"
    assert reindex_payload["chunk_strategy"] == "deterministic"

    db_session.expire_all()
    updated = db_session.query(Document).filter_by(id=ready_document.id).first()
    assert updated.status == "pending"
    assert updated.chunk_strategy == "deterministic"


def test_reindex_rejects_processing_document(
    client, db_session, ready_document, auth_headers
):
    ready_document.status = "processing"
    db_session.commit()

    response = client.post(
        f"/api/v1/documents/{ready_document.id}/reindex",
        json={"strategy": "semantic_llm"},
        headers=auth_headers,
    )

    assert response.status_code == 409


def test_hierarchy_api_hides_other_users_documents(
    client, db_session, other_user, auth_headers
):
    other_doc = Document(
        user_id=other_user.id,
        filename="other.pdf",
        original_name="other.pdf",
        file_size=10,
        status="ready",
    )
    db_session.add(other_doc)
    db_session.commit()
    db_session.refresh(other_doc)

    response = client.get(
        f"/api/v1/documents/{other_doc.id}/hierarchy",
        headers=auth_headers,
    )

    assert response.status_code == 404


# ── Ingestion integration ───────────────────────────────────────────────────


def test_ingest_document_persists_hierarchy_metadata(
    db_session, user, monkeypatch, tmp_path
):
    from app.services import document_ingestion

    document = Document(
        user_id=user.id,
        filename="semantic.pdf",
        original_name="semantic.pdf",
        file_size=1024,
        status="pending",
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    blocks = extract_source_blocks(_elements())
    nodes = build_hierarchy(
        deterministic_sections(blocks), blocks, document_id=str(document.id)
    )
    chunks = generate_chunks(
        nodes,
        blocks,
        document_id=str(document.id),
        document_name="semantic.pdf",
        index_version="semantic-llm-v1",
        max_tokens=1800,
    )
    chunk_list = SemanticChunkList(
        chunks,
        hierarchy=nodes,
        report={"source_coverage": 1.0, "chunk_count": len(chunks)},
        strategy="semantic_llm",
        index_version="semantic-llm-v1",
    )

    fake_pdf = tmp_path / "semantic.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        document_ingestion,
        "materialize_document_file",
        lambda **kwargs: (str(fake_pdf), False),
    )
    monkeypatch.setattr(document_ingestion, "get_page_count", lambda path: 2)
    monkeypatch.setattr(
        document_ingestion, "chunk_document", lambda *args, **kwargs: chunk_list
    )
    monkeypatch.setattr(
        document_ingestion,
        "store_chunks",
        lambda chunks, **kwargs: len(chunks),
    )
    monkeypatch.setattr(document_ingestion, "persist_document_keywords", lambda *a: None)
    monkeypatch.setattr("app.database.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)

    document_ingestion.ingest_document(
        str(document.id), str(fake_pdf), "semantic.pdf", str(user.id)
    )

    db_session.expire_all()
    updated = db_session.query(Document).filter_by(id=document.id).first()
    assert updated.status == "ready"
    assert updated.chunk_strategy == "semantic_llm"
    assert updated.index_version == "semantic-llm-v1"
    assert updated.chunking_report
    assert (
        db_session.query(DocumentSection)
        .filter_by(document_id=str(document.id))
        .count()
        == len(nodes)
    )
    assert (
        db_session.query(DocumentChunk)
        .filter_by(document_id=str(document.id))
        .count()
        == len(chunks)
    )


def test_ingest_document_forwards_per_document_strategy(
    db_session, user, monkeypatch, tmp_path
):
    """A reindex strategy must override the global chunking default."""
    from app.services import document_ingestion

    document = Document(
        user_id=user.id,
        filename="deterministic.pdf",
        original_name="deterministic.pdf",
        file_size=1024,
        status="pending",
        chunk_strategy="deterministic",
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    blocks = extract_source_blocks(_elements())
    nodes = build_hierarchy(
        deterministic_sections(blocks), blocks, document_id=str(document.id)
    )
    chunks = generate_chunks(
        nodes,
        blocks,
        document_id=str(document.id),
        document_name="deterministic.pdf",
        index_version="semantic-fallback-v1",
        max_tokens=1800,
    )
    chunk_list = SemanticChunkList(
        chunks,
        hierarchy=nodes,
        report={"source_coverage": 1.0},
        strategy="deterministic",
        index_version="semantic-fallback-v1",
    )

    fake_pdf = tmp_path / "deterministic.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")
    captured = {}

    def fake_chunk_document(path, **kwargs):
        captured.update(kwargs)
        return chunk_list

    monkeypatch.setattr(
        document_ingestion,
        "materialize_document_file",
        lambda **kwargs: (str(fake_pdf), False),
    )
    monkeypatch.setattr(document_ingestion, "get_page_count", lambda path: 2)
    monkeypatch.setattr(document_ingestion, "chunk_document", fake_chunk_document)
    monkeypatch.setattr(
        document_ingestion,
        "store_chunks",
        lambda chunks, **kwargs: len(chunks),
    )
    monkeypatch.setattr(document_ingestion, "persist_document_keywords", lambda *a: None)
    monkeypatch.setattr("app.database.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)

    document_ingestion.ingest_document(
        str(document.id), str(fake_pdf), "deterministic.pdf", str(user.id)
    )

    assert captured["strategy"] == "deterministic"
    db_session.expire_all()
    updated = db_session.query(Document).filter_by(id=document.id).first()
    assert updated.chunk_strategy == "deterministic"
    assert updated.index_version == "semantic-fallback-v1"
