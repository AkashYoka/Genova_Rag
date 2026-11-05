"""Markdown → ReportLab flowables (shared by Chat export)."""
from __future__ import annotations

import re
from typing import List, Sequence

from reportlab.platypus import Flowable, HRFlowable, Paragraph, Preformatted, Spacer, Table, TableStyle
from reportlab.lib.units import mm

from app.publishing.design import (
    ACCENT,
    BORDER,
    SURFACE,
    TABLE_ALT,
    TABLE_GRID,
    TABLE_HEADER,
)


def safe(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _inline_patterns():
    from app.publishing.fonts import MONO, font

    mono = font(MONO)
    return [
        (re.compile(r"`([^`]+)`"), rf"<font face='{mono}' size='8.5'>\1</font>"),
        (re.compile(r"\*\*\*(.+?)\*\*\*"), r"<b><i>\1</i></b>"),
        (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
        (re.compile(r"__(.+?)__"), r"<b>\1</b>"),
        (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"<i>\1</i>"),
        (re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)"), r"<i>\1</i>"),
        (
            re.compile(r"\[([^\]]+)\]\(([^)]+)\)"),
            r'<link href="\2"><font color="#0F7C86"><u>\1</u></font></link>',
        ),
    ]


_INLINE = None  # lazy via _inline_patterns()



def inline(text: str) -> str:
    global _INLINE
    if _INLINE is None:
        _INLINE = _inline_patterns()
    out = safe(text)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    return out


def _is_table_sep(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)


def _split_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _parse_table(lines: Sequence[str], start: int) -> tuple[List[List[str]], int]:
    rows: List[List[str]] = []
    i = start
    while i < len(lines) and "|" in lines[i]:
        line = lines[i].rstrip()
        if not line.strip():
            break
        if _is_table_sep(line):
            i += 1
            continue
        rows.append(_split_row(line))
        i += 1
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows], i


def make_table(rows: List[List[str]], styles: dict, width: float) -> Table:
    cols = max(len(r) for r in rows) if rows else 1
    col_w = width / cols
    data = []
    for r_i, row in enumerate(rows):
        style = styles["table_header"] if r_i == 0 else styles["table_cell"]
        data.append([Paragraph(inline(cell or " "), style) for cell in row])

    table = Table(
        data,
        colWidths=[col_w] * cols,
        hAlign="LEFT",
        repeatRows=1,
    )
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER),
        ("GRID", (0, 0), (-1, -1), 0.45, TABLE_GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in range(1, len(rows)):
        bg = TABLE_ALT if r % 2 == 0 else SURFACE
        cmds.append(("BACKGROUND", (0, r), (-1, r), bg))
    table.setStyle(TableStyle(cmds))
    return table


def markdown_to_flowables(
    text: str,
    styles: dict,
    content_width: float,
) -> List[Flowable]:
    """Markdown → flowables. Lists use Paragraph glyphs (no ListFlowable overlap)."""
    if not (text or "").strip():
        return [Paragraph("&nbsp;", styles["body"])]

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: List[Flowable] = []
    i = 0
    para: List[str] = []

    def flush() -> None:
        nonlocal para
        if not para:
            return
        joined = " ".join(p.strip() for p in para if p.strip())
        if joined:
            out.append(Paragraph(inline(joined), styles["body"]))
        para = []

    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        if s.startswith("```"):
            flush()
            i += 1
            code: List[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            out.append(Preformatted("\n".join(code) or " ", styles["code"]))
            continue

        if not s:
            flush()
            i += 1
            continue

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", s):
            flush()
            out.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=BORDER,
                    spaceBefore=1.5 * mm,
                    spaceAfter=2.5 * mm,
                )
            )
            i += 1
            continue

        if "|" in s and i + 1 < len(lines) and (
            _is_table_sep(lines[i + 1]) or "|" in lines[i + 1]
        ):
            flush()
            rows, nxt = _parse_table(lines, i)
            if rows:
                out.append(Spacer(1, 1.5 * mm))
                out.append(make_table(rows, styles, content_width))
                out.append(Spacer(1, 2.5 * mm))
            i = nxt
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", s)
        if heading:
            flush()
            level = len(heading.group(1))
            key = {1: "h1", 2: "h2", 3: "h3", 4: "h4", 5: "h5"}.get(level, "h6")
            out.append(Paragraph(inline(heading.group(2)), styles[key]))
            i += 1
            continue

        if re.match(r"^[-*+]\s+", s):
            flush()
            while i < len(lines) and re.match(r"^[-*+]\s+", lines[i].strip()):
                item = re.sub(r"^[-*+]\s+", "", lines[i].strip())
                out.append(Paragraph(f"• {inline(item)}", styles["list"]))
                i += 1
            out.append(Spacer(1, 1 * mm))
            continue

        if re.match(r"^\d+[.)]\s+", s):
            flush()
            n = 1
            while i < len(lines) and re.match(r"^\d+[.)]\s+", lines[i].strip()):
                item = re.sub(r"^\d+[.)]\s+", "", lines[i].strip())
                out.append(Paragraph(f"<b>{n}.</b>  {inline(item)}", styles["list"]))
                n += 1
                i += 1
            out.append(Spacer(1, 1 * mm))
            continue

        if s.startswith(">"):
            flush()
            bits: List[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                bits.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            out.append(Paragraph(inline(" ".join(bits)), styles["quote"]))
            continue

        para.append(s)
        i += 1

    flush()
    return out or [Paragraph("&nbsp;", styles["body"])]
