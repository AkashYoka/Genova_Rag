"""
Entity resolution across documents.

The resolver deliberately favours *not* merging:

  * exact normalised name in the same entity type          → merge
  * candidate name matches a stored alias                  → merge
  * candidate name is an acronym/expansion of a stored one → merge
  * fuzzy name similarity in the same entity type          → keep both, record
    ``possible_duplicate_of`` so a human can review it

Merging happens across documents (the graph is user-scoped), but never across
entity types and never on name similarity alone.
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional

_ACRONYM_STOPWORDS = {"of", "and", "the", "for", "de", "la", "von"}
_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)

MATCH_EXACT = "exact"
MATCH_ALIAS = "alias"
MATCH_ACRONYM = "acronym"
FUZZY_CANDIDATE = "fuzzy_candidate"
NO_MATCH = "new"


def normalize_name(name: str) -> str:
    """Case-fold, strip punctuation and collapse whitespace for comparisons."""
    text = _NON_WORD.sub(" ", (name or "").strip().casefold())
    return " ".join(text.split())


def name_tokens(name: str) -> List[str]:
    return [token for token in normalize_name(name).split() if token]


def acronym(name: str) -> str:
    """Initials of the significant words, e.g. 'Application Backend' → 'ab'."""
    tokens = [token for token in name_tokens(name) if token not in _ACRONYM_STOPWORDS]
    if len(tokens) < 2:
        return ""
    return "".join(token[0] for token in tokens)


def _expand_acronym(name: str) -> str:
    """Uppercase letters run together ("AWSEC2") are treated as an acronym."""
    compact = "".join(name_tokens(name))
    return compact if len(compact) >= 2 else ""


def is_acronym_of(candidate: str, existing: str) -> bool:
    """True when one name is the initialism of the other (or vice versa)."""
    candidate_norm = normalize_name(candidate)
    existing_norm = normalize_name(existing)
    if not candidate_norm or not existing_norm:
        return False
    if candidate_norm == existing_norm:
        return True
    return acronym(existing_norm) == candidate_norm or acronym(candidate_norm) == existing_norm


def similarity(left: str, right: str) -> float:
    """Similarity of two normalised names, with a token-containment bonus."""
    a = normalize_name(left)
    b = normalize_name(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    a_tokens, b_tokens = set(a.split()), set(b.split())
    if a_tokens and b_tokens:
        overlap = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
        if overlap >= 0.5:
            ratio = max(ratio, 0.5 + 0.5 * overlap)
    return ratio


@dataclass
class ExistingEntity:
    id: str
    name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)


@dataclass
class ResolutionDecision:
    action: str  # "match" | "new"
    matched_entity_id: Optional[str] = None
    matched_name: Optional[str] = None
    reason: str = NO_MATCH
    possible_duplicate_of: Optional[str] = None
    new_aliases: List[str] = field(default_factory=list)

    @property
    def merged(self) -> bool:
        return self.action == "match"


def _candidate_aliases(name: str, aliases: Iterable[str]) -> List[str]:
    values = [name, *[alias for alias in aliases or [] if alias]]
    seen = set()
    result = []
    for value in values:
        norm = normalize_name(value)
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def resolve(
    *,
    name: str,
    entity_type: str,
    aliases: Optional[Iterable[str]] = None,
    existing: Iterable[ExistingEntity],
    threshold: float = 0.9,
) -> ResolutionDecision:
    """Decide whether ``name`` refers to an existing entity of the same type."""
    normalized = normalize_name(name)
    if not normalized:
        return ResolutionDecision(action="new", reason=NO_MATCH)

    same_type = [entity for entity in existing if entity.entity_type == entity_type]
    candidate_aliases = _candidate_aliases(name, aliases or [])

    for entity in same_type:
        if normalize_name(entity.name) == normalized:
            extra = [
                alias
                for alias in (aliases or [])
                if alias and alias.casefold() != entity.name.casefold()
            ]
            return ResolutionDecision(
                action="match",
                matched_entity_id=entity.id,
                matched_name=entity.name,
                reason=MATCH_EXACT,
                new_aliases=extra,
            )

    for entity in same_type:
        entity_alias_norms = {normalize_name(alias) for alias in entity.aliases}
        if normalized in entity_alias_norms:
            return ResolutionDecision(
                action="match",
                matched_entity_id=entity.id,
                matched_name=entity.name,
                reason=MATCH_ALIAS,
            )

    for entity in same_type:
        if is_acronym_of(name, entity.name):
            return ResolutionDecision(
                action="match",
                matched_entity_id=entity.id,
                matched_name=entity.name,
                reason=MATCH_ACRONYM,
                new_aliases=[name],
            )

    best_entity: Optional[ExistingEntity] = None
    best_score = 0.0
    for entity in same_type:
        score = max(
            [similarity(name, entity.name)]
            + [similarity(name, alias) for alias in entity.aliases]
        )
        if score > best_score:
            best_score = score
            best_entity = entity

    if best_entity is not None and best_score >= threshold:
        # Similar but unproven: keep the new entity, remember the candidate.
        return ResolutionDecision(
            action="new",
            reason=FUZZY_CANDIDATE,
            possible_duplicate_of=best_entity.id,
        )

    return ResolutionDecision(action="new", reason=NO_MATCH)


def build_lookup(rows: Iterable[Dict]) -> List[ExistingEntity]:
    """Convert stored entity dicts into resolver inputs."""
    lookup: List[ExistingEntity] = []
    for row in rows:
        lookup.append(
            ExistingEntity(
                id=str(row["id"]),
                name=row.get("canonical_name") or row.get("name") or "",
                entity_type=row.get("entity_type") or "",
                aliases=list(row.get("aliases") or []),
            )
        )
    return lookup


__all__ = [
    "ExistingEntity",
    "ResolutionDecision",
    "acronym",
    "build_lookup",
    "is_acronym_of",
    "name_tokens",
    "normalize_name",
    "resolve",
    "similarity",
]
