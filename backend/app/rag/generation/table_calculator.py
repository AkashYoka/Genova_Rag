"""Deterministic multi-cell calculations over SQL-hydrated tables.

Supports operand-cell extraction + sandboxed arithmetic (AST-validated).
Never invents cell values — operands are always read from the hydrated grid.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_NUM_RE = re.compile(r"(?i)(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)")
_UNIT_RE = re.compile(
    r"(?i)\b(mg|g|kg|ml|l|%|iu/?m?l|usd|eur|inr|₹|\$)\b"
)

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass
class OperandCell:
    row_index: int
    col_index: int
    raw_value: str
    numeric_value: Optional[float] = None
    unit: str = ""


@dataclass
class VerifiedCalculation:
    """Deterministic multi-cell calculation result (SQL operands only)."""

    value: Union[float, str]
    operation: str
    operands: List[OperandCell] = field(default_factory=list)
    table_id: str = ""
    page: Optional[int] = None
    pages: List[int] = field(default_factory=list)
    filename: str = ""
    expression: str = ""
    unit: str = ""
    status: str = "calculated"
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        bits = []
        for op in self.operands:
            bits.append(
                f"[{op.row_index},{op.col_index}]={op.raw_value}"
                + (f" (num={op.numeric_value})" if op.numeric_value is not None else "")
            )
        unit = f" {self.unit}" if self.unit else ""
        return (
            "CRITICAL — Verified table calculation is authoritative.\n"
            f"Computed result MUST be used verbatim: {self.value}{unit}.\n"
            f"- Operation: {self.operation}\n"
            f"- Expression: {self.expression or 'n/a'}\n"
            f"- Operand cells (SQL ground truth): {'; '.join(bits)}\n"
            f"- Table id: {self.table_id or 'unknown'}\n"
            f"- Page: {self.page}\n"
            "- Do not invent intermediate values; only use the listed operand cells.\n"
            "- Show step-by-step arithmetic using these exact cell values."
        )

    def to_meta(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "normalized_value": str(self.value),
            "status": "matched",  # reconcile pins like VerifiedCell
            "calculation": True,
            "operation": self.operation,
            "expression": self.expression,
            "unit": self.unit,
            "table_id": self.table_id,
            "page": self.page,
            "pages": list(self.pages or []),
            "row_headers": [f"calc:{self.operation}"],
            "col_headers": ["result"],
            "row_index": None,
            "col_index": None,
            "confidence": 100.0,
            "operands": [
                {
                    "row_index": o.row_index,
                    "col_index": o.col_index,
                    "raw_value": o.raw_value,
                    "numeric_value": o.numeric_value,
                    "unit": o.unit,
                }
                for o in self.operands
            ],
            "evidence": dict(self.evidence or {}),
        }


def parse_numeric_cell(raw: Any) -> Tuple[Optional[float], str, str]:
    """Parse a cell into (number, unit, cleaned_raw). Commas stripped safely."""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None, "", text
    unit_m = _UNIT_RE.search(text)
    unit = unit_m.group(1).lower() if unit_m else ""
    # Prefer the first number-like token; strip thousands separators.
    m = _NUM_RE.search(text.replace(" ", ""))
    if not m:
        m = _NUM_RE.search(text)
    if not m:
        return None, unit, text
    token = m.group(1).replace(",", "")
    try:
        return float(token), unit, text
    except ValueError:
        return None, unit, text


def _safe_eval_arith(expr: str) -> float:
    """Evaluate a numeric arithmetic expression with a strict AST whitelist."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return float(_ALLOWED_UNARY[type(node.op)](_eval(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ZeroDivisionError("division by zero")
            return float(_ALLOWED_BINOPS[type(node.op)](left, right))
        raise ValueError(f"disallowed expression node: {type(node).__name__}")

    return _eval(tree)


def _read_operand(
    table: Dict[str, Any],
    row_index: int,
    col_index: int,
) -> OperandCell:
    rows = [list(r) for r in (table.get("rows") or [])]
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index {row_index} out of bounds")
    row = list(rows[row_index] or [])
    if col_index < 0:
        raise IndexError(f"col_index {col_index} out of bounds")
    if col_index >= len(row):
        row = row + [""] * (col_index - len(row) + 1)
    raw = str(row[col_index] if row[col_index] is not None else "").strip()
    num, unit, _ = parse_numeric_cell(raw)
    return OperandCell(
        row_index=row_index,
        col_index=col_index,
        raw_value=raw,
        numeric_value=num,
        unit=unit,
    )


def _infer_operation(question: str) -> str:
    q = (question or "").lower()
    if any(w in q for w in ("ratio", "divided by", "per ", " / ")):
        return "ratio"
    if any(w in q for w in ("difference", "differ", "minus", "subtract", "delta")):
        return "difference"
    if any(w in q for w in ("sum", "total", "add", "plus", "combined")):
        return "sum"
    if any(w in q for w in ("average", "mean")):
        return "average"
    if any(w in q for w in ("product", "multiply", "times")):
        return "product"
    if any(w in q for w in ("maximum", "highest", "largest", "max ")):
        return "max"
    if any(w in q for w in ("minimum", "lowest", "smallest", "min ")):
        return "min"
    return "calculate"


def compute_from_operands(
    table: Dict[str, Any],
    operands: Sequence[Tuple[int, int]],
    *,
    operation: str = "sum",
    expression: str = "",
    question: str = "",
    table_id: str = "",
    filename: str = "",
) -> VerifiedCalculation:
    """Compute using exact SQL cells at the given coordinates."""
    cells = [_read_operand(table, int(r), int(c)) for r, c in operands]
    nums = [c.numeric_value for c in cells]
    if any(n is None for n in nums):
        raise ValueError(
            "non-numeric operand cell: "
            + ", ".join(
                f"[{c.row_index},{c.col_index}]={c.raw_value!r}" for c in cells
            )
        )
    assert all(n is not None for n in nums)
    values = [float(n) for n in nums]  # type: ignore[arg-type]
    op = (operation or _infer_operation(question) or "sum").lower().strip()

    if expression.strip():
        # Map $0, $1 … to numeric values inside a sandboxed expression.
        expr = expression.strip()
        for i, val in enumerate(values):
            expr = expr.replace(f"${i}", str(val))
            expr = expr.replace(f"{{{i}}}", str(val))
        result = _safe_eval_arith(expr)
        used_expr = expr
    elif op in {"sum", "add", "total"}:
        result = float(sum(values))
        used_expr = " + ".join(str(v) for v in values)
    elif op in {"difference", "subtract", "delta", "diff"}:
        if len(values) < 2:
            raise ValueError("difference requires ≥2 operands")
        result = float(values[0] - values[1])
        used_expr = f"{values[0]} - {values[1]}"
    elif op in {"ratio", "divide"}:
        if len(values) < 2 or values[1] == 0:
            raise ValueError("ratio requires ≥2 operands and non-zero divisor")
        result = float(values[0] / values[1])
        used_expr = f"{values[0]} / {values[1]}"
    elif op in {"product", "multiply"}:
        result = 1.0
        for v in values:
            result *= v
        used_expr = " * ".join(str(v) for v in values)
    elif op in {"average", "mean"}:
        result = float(sum(values) / len(values))
        used_expr = f"({'+'.join(str(v) for v in values)})/{len(values)}"
    elif op in {"max", "maximum"}:
        result = float(max(values))
        used_expr = f"max({', '.join(str(v) for v in values)})"
    elif op in {"min", "minimum"}:
        result = float(min(values))
        used_expr = f"min({', '.join(str(v) for v in values)})"
    else:
        raise ValueError(f"unsupported operation: {op}")

    # Prefer integer formatting when clean.
    if abs(result - round(result)) < 1e-9:
        display: Union[float, int, str] = int(round(result))
    else:
        display = round(result, 6)

    units = [c.unit for c in cells if c.unit]
    unit = units[0] if units and all(u == units[0] for u in units) else ""
    pages = list(table.get("pages") or [])
    page = pages[0] if pages else table.get("page")
    return VerifiedCalculation(
        value=display,
        operation=op,
        operands=cells,
        table_id=table_id or str(table.get("table_id") or ""),
        page=int(page) if page is not None else None,
        pages=[int(p) for p in pages] if pages else [],
        filename=filename,
        expression=used_expr,
        unit=unit,
        evidence={
            "question": question,
            "verification": "python_sql_calculation",
        },
    )


def extract_operand_coords_from_rationale(
    rationale: str,
) -> List[Tuple[int, int]]:
    """Pull ``[row, col]`` mentions from selector rationale text."""
    coords: List[Tuple[int, int]] = []
    for m in re.finditer(r"\[(\d+)\s*,\s*(\d+)\]", rationale or ""):
        coords.append((int(m.group(1)), int(m.group(2))))
    return coords


def try_calculate_from_decision(
    decision: Any,
    table: Dict[str, Any],
    *,
    question: str = "",
    filename: str = "",
) -> Optional[VerifiedCalculation]:
    """Best-effort calculation from a TableLookupDecision + hydrated table.

    Looks for operand coordinates in:
    - ``decision.operands`` (list of {row_index,col_index}) if present
    - ``[r,c]`` mentions in ``decision.rationale``
    Falls back to None when operands cannot be resolved cleanly.
    """
    coords: List[Tuple[int, int]] = []
    raw_ops = getattr(decision, "operands", None)
    if raw_ops is None and isinstance(decision, dict):
        raw_ops = decision.get("operands")
    if isinstance(raw_ops, (list, tuple)):
        for item in raw_ops:
            if isinstance(item, dict):
                try:
                    coords.append((int(item["row_index"]), int(item["col_index"])))
                except (KeyError, TypeError, ValueError):
                    pass
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    coords.append((int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    pass
    if not coords:
        rationale = getattr(decision, "rationale", None)
        if rationale is None and isinstance(decision, dict):
            rationale = decision.get("rationale")
        coords = extract_operand_coords_from_rationale(str(rationale or ""))

    if len(coords) < 2:
        return None

    op = _infer_operation(question)
    table_id = str(
        getattr(decision, "table_id", None)
        or (decision.get("table_id") if isinstance(decision, dict) else "")
        or table.get("table_id")
        or ""
    )
    return compute_from_operands(
        table,
        coords,
        operation=op,
        question=question,
        table_id=table_id,
        filename=filename,
    )


def calculation_fallback_prompt(
    table: Dict[str, Any],
    *,
    question: str,
    table_id: str = "",
    rationale: str = "",
) -> str:
    """Inject full hydrated grid when automated calculation cannot run."""
    from app.rag.table_rag.representations import render_indexed_table_for_selector

    grid = render_indexed_table_for_selector(
        {
            "table_id": table_id or table.get("table_id"),
            "table": table,
            "pages": table.get("pages"),
            "caption": table.get("caption"),
        }
    )
    return (
        "CALCULATION REQUIRED — automated verifier could not resolve operands.\n"
        f"Question: {question}\n"
        f"Selector rationale: {rationale or 'n/a'}\n"
        "Show step-by-step arithmetic using ONLY the exact cell values below. "
        "Do not invent numbers.\n\n"
        f"{grid}\n\n"
    )
