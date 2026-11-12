"""LLM-assisted hierarchical semantic chunking.

The package turns layout-extracted document elements into a section tree with
retrieval-ready chunks. An LLM (Amazon Nova 2 Lite through Bedrock Converse)
identifies section boundaries; chunk text is always reconstructed from the
original extracted blocks so model output can never replace source content.
"""
from app.rag.chunking.semantic.service import (
    semantic_chunk_document,
    should_use_semantic_chunking,
)
from app.rag.chunking.semantic.types import SemanticChunkList

__all__ = [
    "SemanticChunkList",
    "semantic_chunk_document",
    "should_use_semantic_chunking",
]
