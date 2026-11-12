"""Section analysis via Amazon Nova (Bedrock Converse) with strict validation.

The model only returns a hierarchy of section titles and source block ids.
Chunk text is reconstructed from the original blocks afterwards, so a model
mistake can misplace content but never silently lose or rewrite it.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.rag.chunking.semantic.types import (
    AnalysisResult,
    SectionSpec,
    SourceBlock,
)

logger = logging.getLogger(__name__)

MAX_ERRORS = 20

SYSTEM_PROMPT = """You are a document structure analyst for a retrieval-augmented generation system.

Your job is to group the supplied source blocks into a faithful section hierarchy.
Rules:
- You MUST NOT rewrite, summarize, translate, or invent document content.
- You only return section titles, hierarchy levels, and source block ids.
- Every source block id must appear in exactly one section, in reading order.
- A section's block ids must be contiguous and must not overlap another section.
- The heading block that introduces a section must be the first block of that section.
- A table belongs to the section whose text it supports; keep the table block with that section.
- Use level 1 for top-level chapters/major parts, level 2 for subsections, level 3+ for deeper nesting.
- Parent sections must list only their own introductory blocks, never blocks that belong to a child section.
- If the document has no clear headings, infer a small number of topic sections from the content flow.
- Return JSON only. No markdown fences, no commentary.

JSON schema:
{
  "sections": [
    {
      "id": "s1",
      "title": "Section title",
      "level": 1,
      "parent_id": null,
      "block_ids": ["b00001", "b00002"],
      "summary": "One short sentence about this section (optional)."
    }
  ]
}"""


class _LLMSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    title: str = ""
    level: int = 1
    parent_id: Optional[str] = None
    block_ids: List[str] = Field(default_factory=list)
    summary: str = ""


# ── Client construction ─────────────────────────────────────────────────────


def build_semantic_client(settings: Any = None) -> Any:
    """Return an InferenceClient-compatible chat client for section analysis."""
    settings = settings or get_settings()
    provider = (getattr(settings, "SEMANTIC_CHUNKING_PROVIDER", "bedrock") or "bedrock").strip().lower()
    if provider == "ollama":
        from app.rag.llm.ollama_client import OllamaChatClient

        return OllamaChatClient(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL,
        )
    from app.rag.llm.bedrock_client import BedrockNovaChatClient

    model_id = (
        getattr(settings, "SEMANTIC_CHUNKING_MODEL_ID", "")
        or getattr(settings, "BEDROCK_MODEL_ID", "")
        or "us.amazon.nova-2-lite-v1:0"
    )
    return BedrockNovaChatClient(
        model_id=model_id,
        region=getattr(settings, "BEDROCK_REGION", "us-east-1"),
    )


# ── Deterministic fallback ──────────────────────────────────────────────────


def deterministic_sections(
    blocks: Sequence[SourceBlock], *, document_name: str = ""
) -> List[SectionSpec]:
    """Heading-driven hierarchy used when the LLM is unavailable or invalid."""
    specs: List[SectionSpec] = []
    current: Optional[SectionSpec] = None

    for block in blocks:
        if block.is_heading:
            current = SectionSpec(
                title=block.text.strip() or f"Section {len(specs) + 1}",
                level=max(1, min(6, int(block.level or 1))),
                block_ids=[block.block_id],
                source="fallback",
            )
            specs.append(current)
            continue
        if current is None:
            current = SectionSpec(
                title=document_name or "Document",
                level=1,
                block_ids=[],
                source="fallback",
            )
            specs.append(current)
        current.block_ids.append(block.block_id)

    if not specs and blocks:
        specs.append(
            SectionSpec(
                title=document_name or "Document",
                level=1,
                block_ids=[block.block_id for block in blocks],
                source="fallback",
            )
        )
    return specs


# ── JSON parsing ────────────────────────────────────────────────────────────


def _extract_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty LLM response")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S | re.I)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = raw.find(open_ch)
        end = raw.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                continue
    raise ValueError("LLM response did not contain valid JSON")


def _coerce_section(item: Any) -> Optional[_LLMSection]:
    if not isinstance(item, dict):
        return None
    title = (
        item.get("title")
        or item.get("heading")
        or item.get("name")
        or item.get("section")
        or ""
    )
    block_ids = (
        item.get("block_ids")
        or item.get("blocks")
        or item.get("source_block_ids")
        or item.get("blockIds")
        or []
    )
    if isinstance(block_ids, str):
        block_ids = [block_ids]
    if not isinstance(block_ids, (list, tuple)):
        block_ids = []

    level = item.get("level", item.get("depth", 1))
    try:
        level = int(level)
    except Exception:
        level = 1

    parent = item.get("parent_id", item.get("parent", item.get("parentId")))
    summary = item.get("summary", item.get("description", "")) or ""

    return _LLMSection(
        id=str(item.get("id") or item.get("section_id") or ""),
        title=str(title or "").strip(),
        level=max(1, min(6, level)),
        parent_id=str(parent) if parent not in (None, "") else None,
        block_ids=[str(bid).strip() for bid in block_ids if str(bid).strip()],
        summary=str(summary).strip(),
    )


def _parse_llm_sections(payload: Any) -> List[_LLMSection]:
    if isinstance(payload, dict):
        raw_sections = (
            payload.get("sections")
            or payload.get("hierarchy")
            or payload.get("structure")
            or payload.get("outline")
            or []
        )
    elif isinstance(payload, list):
        raw_sections = payload
    else:
        raise ValueError(f"unsupported LLM JSON root: {type(payload).__name__}")

    if isinstance(raw_sections, dict):
        raw_sections = [raw_sections]
    if not isinstance(raw_sections, (list, tuple)):
        raise ValueError("LLM JSON 'sections' must be a list")

    parsed = [_coerce_section(item) for item in raw_sections]
    return [section for section in parsed if section and section.block_ids]


def _to_specs(sections: Iterable[_LLMSection], *, prefix: str) -> List[SectionSpec]:
    specs: List[SectionSpec] = []
    for index, section in enumerate(sections):
        raw_id = section.id or f"s{index + 1}"
        specs.append(
            SectionSpec(
                title=section.title or f"Section {index + 1}",
                level=max(1, min(6, int(section.level or 1))),
                block_ids=list(section.block_ids),
                summary=section.summary,
                ref=f"{prefix}{raw_id}",
                parent_ref=f"{prefix}{section.parent_id}" if section.parent_id else None,
                source="llm",
            )
        )
    return specs


# ── Validation / deterministic repair ───────────────────────────────────────


def validate_and_repair(
    specs: Sequence[SectionSpec],
    blocks: Sequence[SourceBlock],
) -> tuple[List[SectionSpec], Dict[str, Any]]:
    """Guarantee every source block is assigned exactly once.

    Unknown ids are dropped, duplicates keep their first assignment, and any
    unassigned block is appended to the section that precedes it in reading
    order. Content is therefore never lost even with malformed model output.
    """
    valid_ids = {block.block_id for block in blocks}
    order_by_id = {block.block_id: block.order for block in blocks}

    seen: set[str] = set()
    duplicates: List[str] = []
    unknown = 0
    kept: List[SectionSpec] = []

    for spec in specs:
        filtered: List[str] = []
        for block_id in spec.block_ids:
            if block_id not in valid_ids:
                unknown += 1
                continue
            if block_id in seen:
                duplicates.append(block_id)
                continue
            seen.add(block_id)
            filtered.append(block_id)
        if filtered:
            spec.block_ids = filtered
            kept.append(spec)

    if not kept and blocks:
        kept = [
            SectionSpec(
                title="Document",
                level=1,
                block_ids=[],
                source="fallback",
            )
        ]

    missing = [
        block.block_id for block in blocks if block.block_id not in seen
    ]

    if missing and kept:
        # Sort each section's blocks by document order before choosing targets.
        first_order: List[int] = []
        for spec in kept:
            orders = [order_by_id[bid] for bid in spec.block_ids if bid in order_by_id]
            first_order.append(min(orders) if orders else 0)

        for block_id in missing:
            order = order_by_id.get(block_id, 0)
            target = 0
            for index, start in enumerate(first_order):
                if start <= order:
                    target = index
            kept[target].block_ids.append(block_id)
            seen.add(block_id)

    for spec in kept:
        spec.block_ids.sort(key=lambda bid: order_by_id.get(bid, 0))

    stats = {
        "duplicate_block_ids": duplicates[:50],
        "duplicate_count": len(duplicates),
        "unknown_block_ids": unknown,
        "missing_block_count": len(missing),
        "repaired_block_count": len(missing),
        "source_coverage": (len(seen) / len(blocks)) if blocks else 0.0,
    }
    return kept, stats


# ── Windowing ───────────────────────────────────────────────────────────────


def split_windows(
    blocks: Sequence[SourceBlock],
    *,
    max_blocks: int,
    max_tokens: int,
) -> List[List[SourceBlock]]:
    """Split blocks into LLM-sized windows, preferring heading boundaries."""
    if not blocks:
        return []

    windows: List[List[SourceBlock]] = []
    current: List[SourceBlock] = []
    tokens = 0

    for block in blocks:
        cost = block.token_estimate + 24
        if current and (
            len(current) >= max_blocks or tokens + cost > max_tokens
        ):
            windows.append(current)
            current = []
            tokens = 0
        current.append(block)
        tokens += cost

    if current:
        windows.append(current)
    return windows


def _split_point(blocks: Sequence[SourceBlock]) -> int:
    """Pick a heading boundary near the middle of a failed window."""
    if len(blocks) <= 1:
        return len(blocks)
    middle = len(blocks) // 2
    for offset in range(middle):
        for index in (middle + offset, middle - offset):
            if 0 < index < len(blocks) and blocks[index].is_heading:
                return index
    return middle


# ── Analyzer ────────────────────────────────────────────────────────────────


class SemanticAnalyzer:
    """Analyze blocks into section specs, retrying and falling back safely."""

    def __init__(self, *, settings: Any = None, client: Any = None):
        self.settings = settings or get_settings()
        self._client = client
        self._window_seq = 0
        self.llm_calls = 0
        self.retries = 0
        self.used_fallback = False
        self.errors: List[str] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = build_semantic_client(self.settings)
        return self._client

    @property
    def model_id(self) -> str:
        try:
            client = self.client
        except Exception:
            return ""
        return str(
            getattr(client, "model", "")
            or getattr(client, "model_id", "")
            or ""
        )

    def analyze(
        self,
        blocks: Sequence[SourceBlock],
        *,
        document_name: str = "",
        page_count: int = 0,
        progress: Optional[Any] = None,
        strategy: Optional[str] = None,
    ) -> AnalysisResult:
        result = AnalysisResult()
        strategy = (
            strategy
            or getattr(self.settings, "CHUNK_STRATEGY", "semantic_llm")
            or ""
        ).lower()
        enabled = bool(getattr(self.settings, "SEMANTIC_CHUNKING_ENABLED", True))

        if not blocks:
            return result

        if strategy == "deterministic" or not enabled:
            specs = deterministic_sections(blocks, document_name=document_name)
            result.sections = specs
            result.raw_section_count = len(specs)
            return result

        max_blocks = int(
            getattr(self.settings, "SEMANTIC_CHUNKING_MAX_BLOCKS_PER_CALL", 120) or 120
        )
        max_tokens = int(
            getattr(self.settings, "SEMANTIC_CHUNKING_MAX_INPUT_TOKENS", 6000) or 6000
        )
        windows = split_windows(blocks, max_blocks=max_blocks, max_tokens=max_tokens)
        if not windows:
            return result

        all_specs: List[SectionSpec] = []
        total = len(windows)
        for index, window in enumerate(windows):
            if progress:
                try:
                    progress("analyzing", 35 + int((index / max(1, total)) * 20))
                except Exception:
                    pass
            all_specs.extend(
                self._analyze_window(
                    window,
                    document_name=document_name,
                    page_count=page_count,
                    depth=0,
                )
            )

        result.sections = all_specs
        result.raw_section_count = len(all_specs)
        result.used_llm = self.llm_calls > 0
        result.used_fallback = self.used_fallback
        result.windows = self._window_seq
        result.llm_calls = self.llm_calls
        result.retries = self.retries
        result.errors = self.errors[:MAX_ERRORS]
        return result

    # ── internals ──────────────────────────────────────────────────────────

    def _analyze_window(
        self,
        blocks: Sequence[SourceBlock],
        *,
        document_name: str,
        page_count: int,
        depth: int,
    ) -> List[SectionSpec]:
        self._window_seq += 1
        window_id = self._window_seq
        prefix = f"w{window_id}:"

        attempts = max(
            1, int(getattr(self.settings, "SEMANTIC_CHUNKING_RETRIES", 2) or 0) + 1
        )
        feedback = ""
        last_error = ""

        for attempt in range(attempts):
            try:
                payload = self._call_llm(
                    blocks,
                    document_name=document_name,
                    page_count=page_count,
                    window_id=window_id,
                    feedback=feedback,
                )
                sections = _parse_llm_sections(payload)
                specs = _to_specs(sections, prefix=prefix)
                if not specs:
                    raise ValueError("LLM returned no usable sections")
                return specs
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                last_error = f"{type(exc).__name__}: {exc}"
                self.errors.append(
                    f"window {window_id} attempt {attempt + 1}: {last_error}"
                )
                if attempt < attempts - 1:
                    self.retries += 1
                    feedback = (
                        "The previous response was invalid. Return ONLY a JSON object "
                        "matching the schema, with valid source block ids."
                    )

        # Output may have been truncated by the output-token limit. Retry the
        # failed window as smaller sub-windows before giving up on the LLM.
        if len(blocks) > 8 and depth < 3:
            split_at = _split_point(blocks)
            if 0 < split_at < len(blocks):
                left = self._analyze_window(
                    blocks[:split_at],
                    document_name=document_name,
                    page_count=page_count,
                    depth=depth + 1,
                )
                right = self._analyze_window(
                    blocks[split_at:],
                    document_name=document_name,
                    page_count=page_count,
                    depth=depth + 1,
                )
                return left + right

        logger.warning(
            "Semantic analysis fell back to deterministic sections for window %s: %s",
            window_id,
            last_error,
        )
        self.used_fallback = True
        return deterministic_sections(blocks, document_name=document_name)

    def _call_llm(
        self,
        blocks: Sequence[SourceBlock],
        *,
        document_name: str,
        page_count: int,
        window_id: int,
        feedback: str,
    ) -> Any:
        prompt = self._build_user_prompt(
            blocks,
            document_name=document_name,
            page_count=page_count,
            window_id=window_id,
            feedback=feedback,
        )
        self.llm_calls += 1
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.model_id or None,
            max_tokens=int(
                getattr(self.settings, "SEMANTIC_CHUNKING_MAX_OUTPUT_TOKENS", 4096) or 4096
            ),
            temperature=0.0,
            stream=False,
        )
        content = ""
        try:
            content = response.choices[0].message.content or ""
        except Exception:
            content = ""
        return _extract_json(content)

    def _build_user_prompt(
        self,
        blocks: Sequence[SourceBlock],
        *,
        document_name: str,
        page_count: int,
        window_id: int,
        feedback: str,
    ) -> str:
        preview_chars = int(
            getattr(self.settings, "SEMANTIC_CHUNKING_BLOCK_PREVIEW_CHARS", 220) or 220
        )
        lines: List[str] = [
            f"Document: {document_name or 'untitled'}",
            f"Pages: {page_count}",
            f"Block window: {window_id}",
            f"Blocks in this window: {len(blocks)}",
            "",
            "Source blocks (id | type | pages | text preview):",
        ]
        for block in blocks:
            preview = block.preview(preview_chars)
            meta = f"{block.block_id} | {block.element_type}"
            if block.is_heading and block.level:
                meta += f" (heading level {block.level})"
            if block.is_table:
                table = block.table or {}
                headers = table.get("headers") or []
                caption = table.get("caption") or ""
                extra = []
                if caption:
                    extra.append(f"caption={caption[:120]}")
                if headers:
                    extra.append("headers=" + json.dumps(headers[:12], ensure_ascii=False))
                if extra:
                    meta += " | " + "; ".join(extra)
            lines.append(f"{meta} | p{block.page_start}-{block.page_end} | {preview}")
        lines.append("")
        lines.append(
            "Assign EVERY block id above to exactly one section, in reading order. "
            "Return JSON only."
        )
        if feedback:
            lines.append(feedback)
        return "\n".join(lines)


# ── Convenience ─────────────────────────────────────────────────────────────


def analyze_blocks(
    blocks: Sequence[SourceBlock],
    *,
    document_name: str = "",
    page_count: int = 0,
    settings: Any = None,
    client: Any = None,
    progress: Optional[Any] = None,
    strategy: Optional[str] = None,
) -> AnalysisResult:
    analyzer = SemanticAnalyzer(settings=settings, client=client)
    return analyzer.analyze(
        blocks,
        document_name=document_name,
        page_count=page_count,
        progress=progress,
        strategy=strategy,
    )
