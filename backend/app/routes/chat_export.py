"""
PDF export for Genova chat transcripts.

Thin adapter over the shared publishing design system
(`app.publishing`). Preserves the existing `format_pdf` API used by
`chat.py` export endpoints.
"""
from __future__ import annotations

from typing import List

from app.publishing.chat import format_chat_pdf


def format_pdf(doc, messages: List) -> bytes:
    """Render chat history as a professionally typeset study ebook."""
    return format_chat_pdf(doc, messages)
