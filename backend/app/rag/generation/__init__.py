"""Ask-time generation helpers (table lookup evidence, etc.)."""

from app.rag.generation.structured_lookup import (
    VerifiedEvidence,
    build_verified_evidence,
)

__all__ = ["VerifiedEvidence", "build_verified_evidence"]
