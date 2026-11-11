"""LLM-driven table interpretation with strict source-structure validation.

The model receives the complete table (all rows, columns, merged-cell notes and
cell coordinates) and names the table/row/column/cell it used. Every part of
that selection is then checked against the extracted source. A selection that
cannot be proven is reported as uncertainty instead of being turned into an
answer.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.rag.tables.schema import TableRecord, _token_variants
from app.rag.tables.store import tables_from_chunks

logger = logging.getLogger(__name__)

TABLE_EVIDENCE_MARKER = "Verified table evidence"
TABLE_UNCERTAINTY_MARKER = "Table evidence status: UNCERTAIN"
TABLE_CONTEXT_MARKER = "Complete extracted tables"

_STATUS_VERIFIED = "verified"
_STATUS_UNCERTAIN = "uncertain"
_STATUS_NOT_APPLICABLE = "not_applicable"


def _norm_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'`.,;:")


def _parse_identifier(value: Any, prefix: str) -> Optional[int]:
    """Parse ``R3`` / ``r3`` / ``3`` / ``row 3`` into a zero-based index."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value - 1)
    text = _norm_key(value)
    match = re.fullmatch(rf"{prefix}\s*(\d+)", text)
    if match:
        return max(0, int(match.group(1)) - 1)
    if re.fullmatch(r"\d+", text):
        return max(0, int(text) - 1)
    return None


def _parse_cell_identifier(value: Any) -> Optional[Tuple[int, int]]:
    text = _norm_key(value).replace(" ", "")
    match = re.fullmatch(r"r(\d+)c(\d+)", text)
    if not match:
        return None
    return max(0, int(match.group(1)) - 1), max(0, int(match.group(2)) - 1)


def _row_label_keys(record: TableRecord, row: int) -> List[str]:
    keys: List[str] = []
    label = _norm_key(record.row_label(row))
    if label:
        keys.append(label)
    row_text = _norm_key(record.row_text(row))
    if row_text:
        keys.append(row_text)
    for value in record.padded(record.rows[row]) if row < len(record.rows) else []:
        token = _norm_key(value)
        if token:
            keys.append(token)
    return list(dict.fromkeys(keys))


def match_rows(record: TableRecord, value: Any) -> List[int]:
    """Rows that a model-provided label can refer to (empty when unknown)."""
    index = _parse_identifier(value, "r")
    if index is not None:
        return [index] if 0 <= index < record.n_body_rows else []
    key = _norm_key(value)
    if not key:
        return []
    exact: List[int] = []
    partial: List[int] = []
    for row in range(record.n_body_rows):
        keys = _row_label_keys(record, row)
        if key in keys:
            exact.append(row)
        elif any(key in candidate or candidate in key for candidate in keys):
            partial.append(row)
    if exact:
        return exact
    # Prefer a unique partial match; multiple partial matches are ambiguous.
    return partial


def match_columns(record: TableRecord, value: Any) -> List[int]:
    """Columns that a model-provided header label can refer to."""
    index = _parse_identifier(value, "c")
    if index is not None:
        return [index] if 0 <= index < record.n_cols else []
    key = _norm_key(value)
    if not key:
        return []
    exact: List[int] = []
    partial: List[int] = []
    for col in range(record.n_cols):
        label = _norm_key(record.column_label(col))
        leaf_parts = [
            _norm_key(part)
            for part in re.split(r"\s*/\s*|\s*\|\s*", record.column_label(col))
            if _norm_key(part)
        ]
        if key == label or key in leaf_parts:
            exact.append(col)
        elif key in label or label in key:
            partial.append(col)
    if exact:
        return exact
    return partial


def _value_present(cell_text: str, claimed: str) -> bool:
    """True when the model's claimed value exists in the extracted cell."""
    claimed_tokens = _token_variants(claimed)
    cell_tokens = _token_variants(cell_text)
    if claimed_tokens and all(token in cell_tokens for token in claimed_tokens):
        return True
    claimed_key = _norm_key(claimed)
    cell_key = _norm_key(cell_text)
    if claimed_key and claimed_key in cell_key:
        return True
    return False


@dataclass
class TableInterpretation:
    status: str
    records: List[TableRecord] = field(default_factory=list)
    record: Optional[TableRecord] = None
    table_id: str = ""
    row_index: Optional[int] = None
    col_index: Optional[int] = None
    row_label: str = ""
    column_label: str = ""
    cell_id: str = ""
    value: str = ""
    reason: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.status == _STATUS_VERIFIED and self.record is not None

    @property
    def uncertain(self) -> bool:
        return self.status == _STATUS_UNCERTAIN

    def format_block(self) -> str:
        if self.status == _STATUS_NOT_APPLICABLE:
            return ""
        if self.verified and self.record is not None:
            merged = (
                self.record.merged_note(self.row_index, self.col_index)
                if self.row_index is not None and self.col_index is not None
                else ""
            )
            snippet = (
                f"- Table **{self.record.table_id}** (page {self.record.page}"
                + (f", caption: {self.record.caption}" if self.record.caption else "")
                + ")\n"
                f"- Row {self.cell_id.split('C')[0]}: "
                f"{json.dumps(self.row_label, ensure_ascii=False)}\n"
                f"- Column C{(self.col_index or 0) + 1}: "
                f"{json.dumps(self.column_label, ensure_ascii=False)}\n"
                f"- Cell {self.cell_id} value: "
                f"{json.dumps(self.value, ensure_ascii=False)}"
            )
            if merged:
                snippet += f"\n- Merge note: {merged}"
            snippet += (
                f"\n- Source: [Source: {self.record.filename}, Page {self.record.page}]"
            )
            return (
                f"## {TABLE_EVIDENCE_MARKER}\n"
                "The table, row, column and cell below were selected from the "
                "complete extracted table and verified against it. Treat this "
                "cell as the evidence for the table lookup.\n\n"
                f"{snippet}"
            )
        return (
            f"## {TABLE_UNCERTAINTY_MARKER}\n"
            f"Reason: {self.reason or 'table evidence could not be verified'}\n\n"
            "No verified table cell is available. Do not guess a value, row or "
            "column from other excerpts; state clearly that the table evidence "
            "is missing, ambiguous or inconsistent with the question."
        )


def _uncertain(records: Sequence[TableRecord], reason: str, payload: Optional[Dict[str, Any]] = None) -> TableInterpretation:
    return TableInterpretation(
        status=_STATUS_UNCERTAIN,
        records=list(records),
        reason=reason,
        payload=payload or {},
    )


_PLACEHOLDER_TOKEN_RE = re.compile(
    r"^(?:TABLE_ID|TABLE|NULL|NONE|N/?A|C#|R#|R#C#|COLUMN|ROW|EXACT CELL VALUE)$",
    re.I,
)


def _is_placeholder_token(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if _PLACEHOLDER_TOKEN_RE.fullmatch(text):
        return True
    # Schema leftovers such as "C#", "R14C#", "R#C2".
    if "#" in text and re.search(r"[RC]#", text, re.I):
        return True
    return False


def validate_selection(
    records: Sequence[TableRecord],
    payload: Any,
) -> TableInterpretation:
    """Validate a model's table/row/column/cell choice against source data."""
    records = [record for record in records if record is not None]
    if not records:
        return _uncertain([], "no extracted table is available")
    if not isinstance(payload, dict):
        return _uncertain(records, "model returned no structured table selection")

    table_id = str(payload.get("table_id") or payload.get("table") or "").strip()
    if _is_placeholder_token(table_id):
        return _uncertain(
            records,
            f"table id {table_id!r} is a schema placeholder, not an extracted table id",
            payload,
        )
    row_value = (
        payload.get("row")
        if payload.get("row") not in (None, "")
        else payload.get("row_id")
    )
    col_value = (
        payload.get("column")
        if payload.get("column") not in (None, "")
        else payload.get("column_id", payload.get("col"))
    )
    if _is_placeholder_token(row_value) or _is_placeholder_token(col_value):
        return _uncertain(
            records,
            "row/column identifiers are schema placeholders, not real R#/C# values",
            payload,
        )
    cell_value_claim = str(
        payload.get("cell_value")
        or payload.get("value")
        or payload.get("selected_value")
        or ""
    ).strip()
    cell_ref = payload.get("cell_id") or payload.get("cell")
    if _is_placeholder_token(cell_ref):
        return _uncertain(
            records,
            f"cell id {cell_ref!r} is a schema placeholder, not a real R#C# id",
            payload,
        )

    candidates = [record for record in records if record.table_id == table_id] if table_id else list(records)
    if table_id and not candidates:
        ids = ", ".join(record.table_id for record in records)
        return _uncertain(records, f"table id {table_id!r} is not one of the extracted tables ({ids})", payload)
    if not table_id and len(candidates) > 1:
        # No table named: accept only a selection that resolves uniquely.
        pass

    parsed_cell = _parse_cell_identifier(cell_ref) if cell_ref else None
    if cell_ref and parsed_cell is None:
        return _uncertain(records, f"cell id {cell_ref!r} is not a valid R#C# identifier", payload)

    row_index: Optional[int] = None
    col_index: Optional[int] = None
    selected: Optional[TableRecord] = None

    if parsed_cell is not None:
        cell_row, cell_col = parsed_cell

    evaluated: List[Tuple[TableRecord, int, int]] = []
    for record in candidates:
        row_candidates: List[int] = []
        if row_value not in (None, ""):
            row_candidates = match_rows(record, row_value)
        elif parsed_cell is not None:
            row_candidates = [parsed_cell[0]] if parsed_cell[0] < record.n_body_rows else []
        elif record.n_body_rows == 1:
            row_candidates = [0]
        else:
            continue

        col_candidates: List[int] = []
        if col_value not in (None, ""):
            col_candidates = match_columns(record, col_value)
        elif parsed_cell is not None:
            col_candidates = [parsed_cell[1]] if parsed_cell[1] < record.n_cols else []
        elif record.n_cols == 1:
            col_candidates = [0]
        else:
            continue

        for row in row_candidates:
            for col in col_candidates:
                if 0 <= row < record.n_body_rows and 0 <= col < record.n_cols:
                    evaluated.append((record, row, col))

    if not evaluated:
        missing = []
        if row_value in (None, "") and parsed_cell is None:
            missing.append("row")
        if col_value in (None, "") and parsed_cell is None:
            missing.append("column")
        reason = (
            f"the model did not identify a {' and '.join(missing)}"
            if missing
            else f"row {row_value!r} / column {col_value!r} does not exist in the extracted table"
        )
        return _uncertain(records, reason, payload)

    # Deduplicate identical intersections; more than one is genuinely ambiguous.
    unique: List[Tuple[TableRecord, int, int]] = []
    seen = set()
    for record, row, col in evaluated:
        key = (record.table_id, row, col)
        if key not in seen:
            seen.add(key)
            unique.append((record, row, col))

    if len(unique) > 1:
        # Allow the model's explicit cell id to disambiguate a label collision.
        if parsed_cell is not None:
            narrowed = [
                item
                for item in unique
                if item[1] == parsed_cell[0] and item[2] == parsed_cell[1]
            ]
            if len(narrowed) == 1:
                unique = narrowed
        if len(unique) > 1:
            labels = ", ".join(
                f"{record.table_id} {record.cell_id(row, col)}"
                for record, row, col in unique[:5]
            )
            return _uncertain(
                records,
                f"the selection matches multiple cells ({labels}); a single cell cannot be proven",
                payload,
            )

    selected, row_index, col_index = unique[0]

    if parsed_cell is not None and (row_index, col_index) != parsed_cell:
        return _uncertain(
            records,
            f"cell id {cell_ref!r} conflicts with the selected row/column "
            f"({selected.cell_id(row_index, col_index)})",
            payload,
        )

    cell_text = selected.cell_text(row_index, col_index)
    if not cell_text:
        return _uncertain(
            records,
            f"cell {selected.cell_id(row_index, col_index)} is empty in the extracted table",
            payload,
        )

    # When a merged cell is named by a covered coordinate, report the anchor.
    merged = selected.merged_note(row_index, col_index)
    anchor_row, anchor_col = row_index, col_index
    for span in selected.spans:
        try:
            span_row = int(span.get("row", 0))
            span_col = int(span.get("col", 0))
            span_rows = max(1, int(span.get("row_span", 1)))
            span_cols = max(1, int(span.get("col_span", 1)))
        except (TypeError, ValueError):
            continue
        if (
            span_row <= row_index < span_row + span_rows
            and span_col <= col_index < span_col + span_cols
        ):
            anchor_row, anchor_col = span_row, span_col
            break

    if cell_value_claim and not _value_present(cell_text, cell_value_claim):
        return _uncertain(
            records,
            f"the model's value {cell_value_claim!r} does not appear in the "
            f"extracted cell {selected.table_id} "
            f"{selected.cell_id(row_index, col_index)} ({cell_text!r})",
            payload,
        )

    return TableInterpretation(
        status=_STATUS_VERIFIED,
        records=list(records),
        record=selected,
        table_id=selected.table_id,
        row_index=anchor_row,
        col_index=anchor_col,
        row_label=selected.row_label(anchor_row),
        column_label=selected.column_label(anchor_col),
        cell_id=selected.cell_id(anchor_row, anchor_col),
        value=cell_text,
        payload=payload,
    )


def _extract_json_object(content: str) -> Optional[Any]:
    text = (content or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    if text.lower() in {"null", "none", "no table"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def build_table_context(
    records: Sequence[TableRecord],
    *,
    max_chars: Optional[int] = None,
) -> str:
    """Render complete tables for the final answer prompt."""
    records = [record for record in records if record is not None]
    if not records:
        return ""
    budget = max_chars or int(getattr(get_settings(), "TABLE_CONTEXT_MAX_CHARS", 9000) or 9000)
    per_table = max(1200, budget // max(1, len(records)))
    blocks = [record.render_for_prompt(max_chars=per_table) for record in records]
    return f"## {TABLE_CONTEXT_MARKER}\n\n" + "\n\n".join(blocks)


def _interpretation_prompt(question: str, records: Sequence[TableRecord]) -> str:
    rendered = "\n\n".join(
        record.render_for_prompt(max_chars=6000) for record in records
    )
    available_ids = ", ".join(record.table_id for record in records if record.table_id)
    example_id = next((record.table_id for record in records if record.table_id), "tbl_example")
    return (
        "You resolve table evidence for a document question.\n"
        "Use ONLY the tables below. Do not use outside knowledge.\n\n"
        f"Question:\n{question}\n\n"
        f"Available table_id values: {available_ids or '(none)'}\n\n"
        f"Extracted tables:\n{rendered}\n\n"
        "Return exactly one JSON object with these keys:\n"
        f'{{"table_id": "{example_id}", "row": "R3", "column": "C5", '
        '"cell_id": "R3C5", "cell_value": "300", '
        '"reasoning": "brief reason"}}\n\n'
        "Rules:\n"
        "- Copy a real table_id from 'Available table_id values' above. Never write TABLE_ID.\n"
        "- row must be like R1/R2/... and column like C1/C2/... from the printed grid.\n"
        "- cell_value must be copied exactly from that cell (including empty → use null selection).\n"
        "- Select the single cell whose row and column labels match the question.\n"
        "- For a merged cell, report the anchor cell (upper-left R#C#) and its value.\n"
        "- If no table or cell answers the question (for example the answer is in prose, "
        "not a table cell), return {\"table_id\": null}.\n"
        "- Never invent a value, row label or column label.\n"
        "- Output JSON only, with no markdown fences or extra text."
    )


_RANGE_BAND_RE = re.compile(
    r"[>≥]?\s*(\d+(?:\.\d+)?)\s*[-\u2212\u2013\u2014to]+\s*[>≥]?\s*(\d+(?:\.\d+)?)",
    re.I,
)
_TABLE_REF_RE = re.compile(r"\btable\s*([0-9]{1,3})\b", re.I)


def _numbers_in_text(text: str) -> List[float]:
    return [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text or "")]


def _value_in_labeled_band(value: float, label: str) -> bool:
    """True when ``value`` falls inside a numeric range printed in ``label``."""
    for lo_raw, hi_raw in _RANGE_BAND_RE.findall(label or ""):
        try:
            lo = float(lo_raw)
            hi = float(hi_raw)
        except ValueError:
            continue
        if lo > hi:
            lo, hi = hi, lo
        # Bands like ">70-90" mean values above the lower edge through the upper.
        if lo < value <= hi or lo <= value <= hi:
            return True
    return False


def _split_axis_numbers(
    question: str,
) -> Tuple[List[float], List[float]]:
    """Split question numbers into row-axis vs column-axis candidates via local cues.

    Each number is assigned to the nearest generic unit/axis cue within a short
    distance so prescribed dose amounts near ``mg``/``weeks`` do not leak into
    row/column band matching via a wide context window.
    """
    text = question or ""
    cues: List[Tuple[int, int, str]] = []
    for match in re.finditer(r"\b(?:kg|lb|lbs|weight|mass|bw)\b", text, re.I):
        cues.append((match.start(), match.end(), "col"))
    for match in re.finditer(
        r"\b(?:ige|iu|u/?ml|level|serum|assay|index|titer|titre|concentration|ng/?ml|pg/?ml)\b",
        text,
        re.I,
    ):
        cues.append((match.start(), match.end(), "row"))
    for match in re.finditer(
        r"\b(?:mg|mcg|µg|ug|dose|every|week|weeks|q\d)\b",
        text,
        re.I,
    ):
        cues.append((match.start(), match.end(), "skip"))

    row_nums: List[float] = []
    col_nums: List[float] = []
    max_dist = 18
    for match in re.finditer(r"\d+(?:\.\d+)?", text):
        value = float(match.group(0))
        best_kind: Optional[str] = None
        best_dist = max_dist + 1
        for cue_start, cue_end, kind in cues:
            if match.end() <= cue_start:
                dist = cue_start - match.end()
            elif match.start() >= cue_end:
                dist = match.start() - cue_end
            else:
                dist = 0
            if dist < best_dist:
                best_dist = dist
                best_kind = kind
        if best_kind == "col" and best_dist <= max_dist:
            col_nums.append(value)
        elif best_kind == "row" and best_dist <= max_dist:
            row_nums.append(value)
        # skip / uncued numbers are ignored for band matching

    if not row_nums and not col_nums:
        all_nums = _numbers_in_text(question)
        return all_nums, all_nums
    return row_nums, col_nums


def suggest_structural_selection(
    question: str,
    records: Sequence[TableRecord],
) -> Optional[Dict[str, Any]]:
    """Domain-agnostic fallback: map question numbers onto row/column range labels.

    Returns a candidate payload only when exactly one non-empty cell is uniquely
    identified across the provided tables. Prefer a table whose caption number
    matches an explicit ``Table N`` reference in the question when present.
    """
    records = [record for record in records if record is not None and record.rows]
    if not records:
        return None
    row_numbers, col_numbers = _split_axis_numbers(question)
    if not row_numbers or not col_numbers:
        return None

    preferred_table_num: Optional[str] = None
    table_ref = _TABLE_REF_RE.search(question or "")
    if table_ref:
        preferred_table_num = table_ref.group(1)

    hits: List[Tuple[TableRecord, int, int, str]] = []
    for record in records:
        row_hits: List[int] = []
        labels = record.row_labels or [
            " ".join(str(cell) for cell in row[:1]) for row in record.rows
        ]
        for index, label in enumerate(labels):
            if any(_value_in_labeled_band(number, label) for number in row_numbers):
                row_hits.append(index)
        col_hits: List[int] = []
        for index, header in enumerate(record.headers or []):
            if index == 0 and len(record.headers) > 2:
                # Leading label columns rarely carry the measured value.
                continue
            if any(_value_in_labeled_band(number, header) for number in col_numbers):
                col_hits.append(index)
        if not row_hits or not col_hits:
            continue
        for row in row_hits:
            for col in col_hits:
                if row >= len(record.rows) or col >= len(record.rows[row]):
                    continue
                value = str(record.rows[row][col] or "").strip()
                if not value:
                    continue
                hits.append((record, row, col, value))

    if preferred_table_num:
        preferred_hits = [
            hit
            for hit in hits
            if re.search(
                rf"\btable\s*{preferred_table_num}\b",
                str(hit[0].caption or ""),
                re.I,
            )
        ]
        if preferred_hits:
            hits = preferred_hits

    if len(hits) != 1:
        return None
    record, row, col, value = hits[0]
    return {
        "table_id": record.table_id,
        "row": f"R{row + 1}",
        "column": f"C{col + 1}",
        "cell_id": f"R{row + 1}C{col + 1}",
        "cell_value": value,
        "reasoning": "structural range match from question numbers to row/column labels",
    }


def _payload_from_model_content(
    content: str,
    records: Sequence[TableRecord],
) -> TableInterpretation:
    payload = _extract_json_object(content)
    if payload is None:
        return _uncertain(records, "the model returned no parseable table selection")
    if isinstance(payload, dict) and not str(payload.get("table_id") or "").strip() and not any(
        payload.get(key) not in (None, "")
        for key in ("row", "row_id", "column", "column_id", "cell_id", "cell")
    ):
        return TableInterpretation(
            status=_STATUS_NOT_APPLICABLE,
            records=list(records),
            payload=payload,
        )
    return validate_selection(records, payload)


def interpret_table_question(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    client: Any = None,
    model: Optional[str] = None,
    llm_provider: Optional[str] = None,
    max_tables: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> Optional[TableInterpretation]:
    """Ask the configured model to select evidence, then validate it strictly.

    Returns ``None`` when no table exists or the model call itself failed (the
    caller then falls back to the legacy deterministic lookup). Returns a
    ``TableInterpretation`` with status ``not_applicable`` when the model
    reports that no table answers the question.
    """
    settings = get_settings()
    limit = int(
        max_tables
        if max_tables is not None
        else getattr(settings, "TABLE_MAX_CANDIDATES", 3) or 3
    )
    records = tables_from_chunks(chunks, max_tables=limit)
    if not records:
        return None

    if client is None:
        try:
            from app.rag.llm.agent import get_llm_client

            client = get_llm_client(llm_provider=llm_provider)
        except Exception as exc:
            logger.warning("Table interpretation skipped: no LLM client (%s)", exc)
            return None

    resolved_model = model or getattr(settings, "LLM_MODEL", "") or ""
    token_budget = int(
        max_tokens
        or getattr(settings, "TABLE_INTERPRETATION_MAX_TOKENS", 700)
        or 700
    )

    def _call(prompt: str) -> str:
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a table evidence selector. You output only a "
                        "single valid JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=resolved_model,
            max_tokens=token_budget,
            temperature=0.0,
        )
        try:
            if getattr(response, "choices", None):
                return response.choices[0].message.content or ""
        except Exception:
            return ""
        return ""

    try:
        content = _call(_interpretation_prompt(question, records))
    except Exception as exc:
        logger.warning("Table interpretation call failed: %s", exc)
        return None

    interpretation = _payload_from_model_content(content, records)

    # Small local models often echo schema placeholders; give one repair pass.
    if interpretation.status == _STATUS_UNCERTAIN and (
        "placeholder" in (interpretation.reason or "").lower()
        or "no parseable" in (interpretation.reason or "").lower()
    ):
        available_ids = ", ".join(record.table_id for record in records)
        repair_prompt = (
            "Your previous JSON was invalid for this task.\n"
            f"Previous output:\n{content}\n\n"
            f"Valid table_id values: {available_ids}\n"
            "Return corrected JSON only, using a real table_id and real R#/C# "
            "identifiers from the tables. If no cell applies, return "
            '{"table_id": null}.'
        )
        try:
            repaired = _call(repair_prompt)
            interpretation = _payload_from_model_content(repaired, records)
        except Exception as exc:
            logger.debug("Table interpretation repair call failed: %s", exc)

    # Structural numeric-band fallback:
    # - when the model selection is invalid/uncertain
    # - or when the question names Table N but the model picked a different table
    # Never overrides an explicit not_applicable.
    table_ref = _TABLE_REF_RE.search(question or "")
    preferred_table_num = table_ref.group(1) if table_ref else None
    model_missed_named_table = False
    if (
        interpretation.status == _STATUS_VERIFIED
        and preferred_table_num
        and interpretation.record is not None
    ):
        caption = str(interpretation.record.caption or "")
        if re.search(rf"\btable\s*{preferred_table_num}\b", caption, re.I) is None:
            model_missed_named_table = True

    if interpretation.status == _STATUS_UNCERTAIN or model_missed_named_table:
        suggested = suggest_structural_selection(question, records)
        if suggested is not None:
            structural = validate_selection(records, suggested)
            if structural.status == _STATUS_VERIFIED:
                structural.reason = (
                    (structural.reason or "")
                    + (" " if structural.reason else "")
                    + (
                        "selected via structural range match after invalid model JSON"
                        if interpretation.status == _STATUS_UNCERTAIN
                        else "selected via structural range match for the named table"
                    )
                ).strip()
                interpretation = structural

    logger.info(
        "TABLE_FLOW %s",
        json.dumps(
            {
                "event": "table_interpretation",
                "status": interpretation.status,
                "candidate_table_ids": [record.table_id for record in records],
                "selected_table_id": interpretation.table_id,
                "selected_cell": interpretation.cell_id,
                "selected_value": interpretation.value,
                "reason": interpretation.reason,
                "payload": interpretation.payload,
            },
            ensure_ascii=False,
        ),
    )
    return interpretation
