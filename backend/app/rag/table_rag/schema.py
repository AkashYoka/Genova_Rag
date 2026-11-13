"""Pydantic schemas for LLM table understanding metadata.

LLM output is interpretation metadata only. Original extracted cell values,
headers, merged cells, and bboxes remain the source of truth elsewhere.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ColumnMeaning(BaseModel):
    model_config = ConfigDict(extra="ignore")

    header: str = ""
    meaning: str = ""
    unit: str = ""
    is_axis: bool = False
    is_lookup_value: bool = False
    notes: str = ""


class RowRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str = ""
    related_headers: List[str] = Field(default_factory=list)


class HierarchicalHeaderNote(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: int = 1
    text: str = ""
    spans_columns: List[str] = Field(default_factory=list)
    interpretation: str = ""


class LookupAttribute(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    role: str = ""  # row_axis | col_axis | value | frequency | other
    header_match: str = ""
    example_values: List[str] = Field(default_factory=list)
    notes: str = ""


class SuggestedQAPattern(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_pattern: str = ""
    answer_strategy: str = ""
    required_axes: List[str] = Field(default_factory=list)


class FootnoteExplanation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    marker: str = ""
    explanation: str = ""
    applies_to: str = ""


class ApplicabilityAttribute(BaseModel):
    """Domain-agnostic key/value applicability metadata (not hardcoded fields).

    Examples (illustrative only — values come from the table text):
    ``{"key": "population", "value": "ages 12 years and older"}``,
    ``{"key": "reporting_period", "value": "FY2024"}``,
    ``{"key": "material_grade", "value": "ASTM A36"}``.
    """

    model_config = ConfigDict(extra="ignore")

    key: str = ""
    value: str = ""
    notes: str = ""
    # explicit = stated in caption/section; inferred = LLM guess; conflict flagged
    evidence_type: str = "explicit"  # explicit | inferred | unknown
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    provenance: str = ""  # caption | section | llm | surrounding_context


class TableUnderstanding(BaseModel):
    """Validated LLM interpretation of one extracted table.

    Never contains authoritative cell values that replace extraction.
    """

    model_config = ConfigDict(extra="ignore")

    table_id: str = ""
    block_id: str = ""
    description: str = ""
    purpose: str = ""
    subject: str = ""
    # Free-text cues that distinguish this table from sibling matrices
    # (population, period, category, scenario). Domain-agnostic strings.
    applicability_cues: List[str] = Field(default_factory=list)
    applicability_attributes: List[ApplicabilityAttribute] = Field(
        default_factory=list
    )
    # Dimensions that must be known to choose among sibling tables.
    differentiating_dimensions: List[str] = Field(default_factory=list)
    column_meanings: List[ColumnMeaning] = Field(default_factory=list)
    hierarchical_headers: List[HierarchicalHeaderNote] = Field(default_factory=list)
    row_column_relationships: List[RowRelationship] = Field(default_factory=list)
    units_and_measurements: List[str] = Field(default_factory=list)
    footnotes: List[FootnoteExplanation] = Field(default_factory=list)
    semantic_search_phrases: List[str] = Field(default_factory=list)
    suggested_qa_patterns: List[SuggestedQAPattern] = Field(default_factory=list)
    lookup_attributes: List[LookupAttribute] = Field(default_factory=list)
    numerical_dimensions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertain: bool = False
    uncertainty_notes: List[str] = Field(default_factory=list)
    # Audit fields filled by validation — not LLM-authored.
    validated: bool = False
    validation_warnings: List[str] = Field(default_factory=list)
    source: str = "llm"  # llm | deterministic_fallback

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, conf))

    def to_index_text(self, *, max_phrases: int = 12) -> str:
        """Compact semantic text for embedding enrichment (not cell truth)."""
        parts: List[str] = ["LLM TABLE INTERPRETATION (not cell ground truth):"]
        if self.description:
            parts.append(f"description: {self.description}")
        if self.purpose:
            parts.append(f"purpose: {self.purpose}")
        if self.subject:
            parts.append(f"subject: {self.subject}")
        if self.applicability_cues:
            parts.append(
                "applicability: " + "; ".join(self.applicability_cues[:12])
            )
        if self.applicability_attributes:
            parts.append(
                "applicability_attrs: "
                + "; ".join(
                    f"{a.key}={a.value}"
                    for a in self.applicability_attributes[:12]
                    if a.key or a.value
                )
            )
        if self.differentiating_dimensions:
            parts.append(
                "differentiating_dimensions: "
                + "; ".join(self.differentiating_dimensions[:8])
            )
        if self.column_meanings:
            bits = []
            for col in self.column_meanings[:16]:
                unit = f" ({col.unit})" if col.unit else ""
                bits.append(f"{col.header}: {col.meaning}{unit}".strip(": "))
            parts.append("columns: " + "; ".join(b for b in bits if b.strip(": ")))
        if self.hierarchical_headers:
            parts.append(
                "header_hierarchy: "
                + "; ".join(
                    f"L{h.level} {h.text} → {h.interpretation}".strip(" →")
                    for h in self.hierarchical_headers[:8]
                )
            )
        if self.row_column_relationships:
            parts.append(
                "relationships: "
                + "; ".join(r.description for r in self.row_column_relationships[:8] if r.description)
            )
        if self.units_and_measurements:
            parts.append("units: " + "; ".join(self.units_and_measurements[:8]))
        if self.footnotes:
            parts.append(
                "footnotes: "
                + "; ".join(
                    f"{f.marker} {f.explanation}".strip()
                    for f in self.footnotes[:6]
                )
            )
        if self.lookup_attributes:
            parts.append(
                "lookup_axes: "
                + "; ".join(
                    f"{a.name}[{a.role}]~{a.header_match}"
                    for a in self.lookup_attributes[:10]
                )
            )
        if self.numerical_dimensions:
            parts.append(
                "numerical_dimensions: " + "; ".join(self.numerical_dimensions[:10])
            )
        phrases = [p for p in self.semantic_search_phrases if p.strip()][:max_phrases]
        if phrases:
            parts.append("search_phrases:")
            parts.extend(f"- {p}" for p in phrases)
        if self.suggested_qa_patterns:
            parts.append("qa_patterns:")
            for qa in self.suggested_qa_patterns[:6]:
                parts.append(f"- Q: {qa.question_pattern} → {qa.answer_strategy}")
        if self.uncertain or self.validation_warnings:
            parts.append(
                "interpretation_warning: verify against extracted table_json cells"
            )
        return "\n".join(parts).strip()

    def to_storage_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TableUnderstandingBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tables: List[TableUnderstanding] = Field(default_factory=list)
