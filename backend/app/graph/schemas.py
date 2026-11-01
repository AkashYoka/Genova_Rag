"""
Typed, validated shapes for knowledge-graph extraction.

Neither entity nor relationship types are enumerated here: whatever the model
finds in the document is normalised into a stable casing convention and stored.
Anything that fails validation is dropped and counted, never persisted.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MAX_NAME_CHARS = 180
MAX_DESCRIPTION_CHARS = 600
MAX_EVIDENCE_CHARS = 700
MAX_ALIASES = 6
MAX_ENTITY_TYPE_CHARS = 48
MAX_RELATION_TYPE_CHARS = 48


class GraphExtractionError(Exception):
    """Raised when extraction could not produce a usable payload at all."""


@dataclass
class ExtractedEntity:
    key: str
    name: str
    entity_type: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    confidence: float = 0.0
    chunk_id: str = ""
    page: Optional[int] = None
    section: str = ""
    evidence: str = ""
    extraction_method: str = "llm"


@dataclass
class ExtractedRelationship:
    source_key: str
    target_key: str
    relation_type: str
    description: str = ""
    confidence: float = 0.0
    chunk_id: str = ""
    page: Optional[int] = None
    section: str = ""
    evidence: str = ""
    extraction_method: str = "llm"


@dataclass
class ExtractionResult:
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    rejected_entities: int = 0
    rejected_relationships: int = 0
    chunks_sent: int = 0
    batches_attempted: int = 0
    failed_batches: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.entities and not self.relationships


def clean_text(value: Any, limit: int) -> str:
    """Collapse whitespace and hard-truncate any model-provided string."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:limit]


def clamp_confidence(value: Any) -> float:
    """Coerce a model-reported confidence into 0..1, defaulting to 0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    if number > 1.0:
        # Some models answer on a 0-100 scale.
        number = number / 100.0
    return max(0.0, min(1.0, number))


def normalize_entity_type(value: Any) -> str:
    """Normalise a free-text entity type to UPPER_SNAKE_CASE."""
    text = clean_text(value, MAX_ENTITY_TYPE_CHARS).upper()
    text = "".join(char if char.isalnum() else "_" for char in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "CONCEPT"


def normalize_relation_type(value: Any) -> str:
    """Normalise a free-text relationship label to lower_snake_case."""
    text = clean_text(value, MAX_RELATION_TYPE_CHARS).lower()
    text = "".join(char if char.isalnum() else "_" for char in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "related_to"


def normalize_aliases(values: Any, canonical: str) -> List[str]:
    if not isinstance(values, list):
        return []
    seen = {canonical.casefold()}
    aliases: List[str] = []
    for value in values:
        alias = clean_text(value, MAX_NAME_CHARS)
        lowered = alias.casefold()
        if not alias or lowered in seen:
            continue
        seen.add(lowered)
        aliases.append(alias)
        if len(aliases) >= MAX_ALIASES:
            break
    return aliases


def _entity_key(entity: Dict[str, Any], index: int) -> str:
    key = clean_text(entity.get("key") or entity.get("id") or "", 80)
    if key:
        return key
    name = clean_text(entity.get("name") or entity.get("entity"), MAX_NAME_CHARS)
    return f"auto:{index}:{name.casefold()}" if name else f"auto:{index}"


def _chunk_meta(chunk_lookup: Dict[str, Dict[str, Any]], chunk_id: str) -> Dict[str, Any]:
    return chunk_lookup.get(chunk_id, {}) if chunk_id else {}


def parse_extraction_payload(
    payload: Any,
    chunk_lookup: Dict[str, Dict[str, Any]],
    *,
    min_confidence: float = 0.0,
    default_chunk_id: str = "",
) -> ExtractionResult:
    """Validate one model response into extraction dataclasses.

    Validation rules (all enforced here, not at the storage layer):
      * entities need a name and type; types are normalised, not restricted
      * entities/relationships below ``min_confidence`` are dropped
      * relationships must reference a known entity *and* carry evidence text
      * relationships without evidence are never invented from co-occurrence
    """
    result = ExtractionResult()
    if not isinstance(payload, dict):
        raise GraphExtractionError("Extraction payload is not a JSON object")

    raw_entities = payload.get("entities")
    raw_relationships = payload.get("relationships")

    by_key: Dict[str, ExtractedEntity] = {}
    for index, raw in enumerate(raw_entities if isinstance(raw_entities, list) else []):
        if not isinstance(raw, dict):
            result.rejected_entities += 1
            continue

        name = clean_text(raw.get("name") or raw.get("entity"), MAX_NAME_CHARS)
        if not name:
            result.rejected_entities += 1
            continue
        entity_type = normalize_entity_type(raw.get("type") or raw.get("entity_type"))
        confidence = clamp_confidence(raw.get("confidence"))
        if confidence and confidence < min_confidence:
            result.rejected_entities += 1
            continue

        chunk_id = clean_text(raw.get("chunk_id"), 128) or default_chunk_id
        meta = _chunk_meta(chunk_lookup, chunk_id)
        entity = ExtractedEntity(
            key=_entity_key(raw, index),
            name=name,
            entity_type=entity_type,
            description=clean_text(raw.get("description"), MAX_DESCRIPTION_CHARS),
            aliases=normalize_aliases(raw.get("aliases"), name),
            confidence=confidence or max(min_confidence, 0.5),
            chunk_id=chunk_id,
            page=meta.get("page"),
            section=clean_text(raw.get("section") or meta.get("section"), 160),
            evidence=clean_text(raw.get("evidence"), MAX_EVIDENCE_CHARS),
        )
        # Merge repeats of the same entity inside one response.
        existing = by_key.get(entity.key)
        if existing is None:
            by_key[entity.key] = entity
        else:
            existing.confidence = max(existing.confidence, entity.confidence)
            if not existing.description and entity.description:
                existing.description = entity.description
            for alias in entity.aliases:
                if alias not in existing.aliases and len(existing.aliases) < MAX_ALIASES:
                    existing.aliases.append(alias)

    if not by_key:
        # Fall back to matching relationship endpoints against entity names.
        for raw in raw_entities if isinstance(raw_entities, list) else []:
            if isinstance(raw, dict) and raw.get("name"):
                by_key[clean_text(raw["name"], MAX_NAME_CHARS).casefold()] = None  # type: ignore[assignment]

    key_lookup = {key: entity for key, entity in by_key.items() if entity is not None}
    name_lookup = {entity.name.casefold(): entity for entity in key_lookup.values()}

    seen_relationships = set()
    for raw in raw_relationships if isinstance(raw_relationships, list) else []:
        if not isinstance(raw, dict):
            result.rejected_relationships += 1
            continue

        source_key = clean_text(raw.get("source") or raw.get("from"), 120)
        target_key = clean_text(raw.get("target") or raw.get("to"), 120)
        source = key_lookup.get(source_key) or name_lookup.get(source_key.casefold())
        target = key_lookup.get(target_key) or name_lookup.get(target_key.casefold())
        if source is None or target is None or source.key == target.key:
            result.rejected_relationships += 1
            continue

        relation_type = normalize_relation_type(raw.get("type") or raw.get("relation"))
        evidence = clean_text(raw.get("evidence"), MAX_EVIDENCE_CHARS)
        if not evidence:
            # No supporting passage → not a relationship we can defend.
            result.rejected_relationships += 1
            continue

        confidence = clamp_confidence(raw.get("confidence"))
        if confidence and confidence < min_confidence:
            result.rejected_relationships += 1
            continue

        chunk_id = clean_text(raw.get("chunk_id"), 128) or source.chunk_id or default_chunk_id
        meta = _chunk_meta(chunk_lookup, chunk_id)
        dedupe_key = (source.key, target.key, relation_type)
        if dedupe_key in seen_relationships:
            continue
        seen_relationships.add(dedupe_key)

        result.relationships.append(
            ExtractedRelationship(
                source_key=source.key,
                target_key=target.key,
                relation_type=relation_type,
                description=clean_text(raw.get("description"), MAX_DESCRIPTION_CHARS),
                confidence=confidence or max(min_confidence, 0.5),
                chunk_id=chunk_id,
                page=meta.get("page"),
                section=clean_text(raw.get("section") or meta.get("section"), 160),
                evidence=evidence,
            )
        )

    result.entities = list(key_lookup.values())
    return result
