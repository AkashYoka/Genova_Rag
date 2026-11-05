"""CKEditor HTML → ReportLab flowables.

Preserves heading hierarchy, lists (nested), tables, images, links,
blockquotes, code, and inline formatting without screenshotting the editor.
"""
from __future__ import annotations

import base64
import re
from html.parser import HTMLParser
from io import BytesIO
from typing import List, Optional, Tuple
from urllib.request import urlopen

from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

from app.publishing.design import BORDER
from app.publishing.markdown import make_table, safe


_BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "blockquote", "pre", "table",
    "tr", "td", "th", "thead", "tbody", "figure", "figcaption",
    "hr", "br",
}
_SKIP_TAGS = {"script", "style", "head", "meta", "link", "iframe"}
_INLINE_TAGS = {
    "b", "strong", "i", "em", "u", "s", "strike", "code", "a",
    "span", "sub", "sup", "mark", "small",
}


def heading_bookmark_name(title: str, index: int) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", (title or "").lower()).strip("-")[:40]
    return f"h-{index}-{base or 'section'}"


class _HtmlToFlowables(HTMLParser):
    def __init__(self, styles: dict, content_width: float):
        super().__init__(convert_charrefs=True)
        self.styles = styles
        self.content_width = content_width
        self.flows: List[Flowable] = []
        self.heading_bookmarks: List[Tuple[str, str, int]] = []  # name, title, level
        self._heading_count = 0
        self._stack: List[str] = []
        self._skip_depth = 0
        self._buf: List[str] = []
        self._list_stack: List[Tuple[str, int]] = []  # (ul|ol, counter)
        self._in_pre = False
        self._pre_buf: List[str] = []
        self._in_table = False
        self._table_rows: List[List[str]] = []
        self._table_row: List[str] = []
        self._cell_buf: List[str] = []
        self._in_cell = False
        self._link_href: Optional[str] = None
        self._inline_open: List[str] = []

    # ── helpers ──────────────────────────────────────────────────────────────
    def _flush_paragraph(self, style_key: str = "body") -> None:
        text = "".join(self._buf).strip()
        self._buf = []
        if not text:
            return
        # Collapse whitespace for normal paragraphs
        if style_key != "code":
            text = re.sub(r"\s+", " ", text)
        if style_key in ("h1", "h2", "h3", "h4", "h5", "h6"):
            from app.publishing.builder import BookmarkAnchor
            import html as html_lib

            plain = re.sub(r"<[^>]+>", "", html_lib.unescape(text)).strip()
            self._heading_count += 1
            level = int(style_key[1])
            name = heading_bookmark_name(plain, self._heading_count)
            self.heading_bookmarks.append((name, plain, level))
            self.flows.append(BookmarkAnchor(name, plain, max(0, level - 1)))
        self.flows.append(Paragraph(text, self.styles[style_key]))

    def _push_inline(self, tag: str, attrs: dict) -> None:
        if tag in ("b", "strong"):
            self._buf.append("<b>")
            self._inline_open.append("</b>")
        elif tag in ("i", "em"):
            self._buf.append("<i>")
            self._inline_open.append("</i>")
        elif tag == "u":
            self._buf.append("<u>")
            self._inline_open.append("</u>")
        elif tag in ("s", "strike"):
            self._buf.append("<strike>")
            self._inline_open.append("</strike>")
        elif tag == "code":
            from app.publishing.fonts import MONO, font

            self._buf.append(f"<font face='{font(MONO)}' size='8.5'>")
            self._inline_open.append("</font>")
        elif tag == "sub":
            self._buf.append("<sub>")
            self._inline_open.append("</sub>")
        elif tag == "sup":
            self._buf.append("<super>")
            self._inline_open.append("</super>")
        elif tag == "a":
            href = attrs.get("href") or ""
            self._link_href = href
            if href:
                self._buf.append(f'<link href="{safe(href)}"><font color="#0F7C86"><u>')
                self._inline_open.append("</u></font></link>")
            else:
                self._inline_open.append("")
        elif tag == "span":
            self._inline_open.append("")
        else:
            self._inline_open.append("")

    def _pop_inline(self) -> None:
        if self._inline_open:
            close = self._inline_open.pop()
            if close:
                self._buf.append(close)

    def _emit_list_item(self) -> None:
        text = "".join(self._buf).strip()
        self._buf = []
        if not text:
            return
        text = re.sub(r"\s+", " ", text)
        depth = max(0, len(self._list_stack) - 1)
        style = self.styles["list_nested"] if depth > 0 else self.styles["list"]
        kind, counter = self._list_stack[-1] if self._list_stack else ("ul", 0)
        if kind == "ol":
            self.flows.append(Paragraph(f"<b>{counter}.</b>  {text}", style))
        else:
            bullet = "◦" if depth > 0 else "•"
            self.flows.append(Paragraph(f"{bullet} {text}", style))

    def _finish_table(self) -> None:
        if not self._table_rows:
            return
        # Normalize columns
        width = max(len(r) for r in self._table_rows)
        rows = [r + [""] * (width - len(r)) for r in self._table_rows]
        self.flows.append(Spacer(1, 1.5 * mm))
        self.flows.append(make_table(rows, self.styles, self.content_width))
        self.flows.append(Spacer(1, 2.5 * mm))
        self._table_rows = []

    def _load_image(self, src: str) -> Optional[Image]:
        if not src:
            return None
        try:
            data: bytes
            if src.startswith("data:"):
                # data:image/png;base64,...
                _, b64 = src.split(",", 1)
                data = base64.b64decode(b64)
            elif src.startswith(("http://", "https://")):
                with urlopen(src, timeout=8) as resp:  # nosec - user content
                    data = resp.read()
            else:
                return None
            max_w = self.content_width
            max_h = 120 * mm
            img = Image(BytesIO(data))
            iw, ih = float(img.imageWidth), float(img.imageHeight)
            if iw <= 0 or ih <= 0:
                return None
            scale = min(max_w / iw, max_h / ih, 1.0)
            img.drawWidth = iw * scale
            img.drawHeight = ih * scale
            img.hAlign = "CENTER"
            return img
        except Exception:
            return None

    # ── parser callbacks ─────────────────────────────────────────────────────
    def handle_starttag(self, tag: str, attrs_list) -> None:
        tag = tag.lower()
        attrs = dict(attrs_list or [])
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if tag == "br":
            self._buf.append("<br/>")
            return

        if tag == "hr":
            self._flush_paragraph()
            self.flows.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=BORDER,
                    spaceBefore=2 * mm,
                    spaceAfter=3 * mm,
                )
            )
            return

        if tag == "img":
            self._flush_paragraph()
            img = self._load_image(attrs.get("src") or "")
            if img:
                alt = attrs.get("alt") or attrs.get("title") or ""
                parts: List[Flowable] = [Spacer(1, 2 * mm), img]
                if alt:
                    parts.append(Paragraph(safe(alt), self.styles["caption"]))
                else:
                    parts.append(Spacer(1, 2 * mm))
                self.flows.append(KeepTogether(parts))
            return

        if tag in ("ul", "ol"):
            self._flush_paragraph()
            self._list_stack.append((tag, 0))
            self._stack.append(tag)
            return

        if tag == "li":
            self._buf = []
            if self._list_stack:
                kind, n = self._list_stack[-1]
                self._list_stack[-1] = (kind, n + 1)
            self._stack.append(tag)
            return

        if tag == "pre":
            self._flush_paragraph()
            self._in_pre = True
            self._pre_buf = []
            self._stack.append(tag)
            return

        if tag == "blockquote":
            self._flush_paragraph()
            self._stack.append(tag)
            return

        if tag == "table":
            self._flush_paragraph()
            self._in_table = True
            self._table_rows = []
            self._stack.append(tag)
            return

        if tag == "tr":
            self._table_row = []
            self._stack.append(tag)
            return

        if tag in ("td", "th"):
            self._in_cell = True
            self._cell_buf = []
            self._stack.append(tag)
            return

        if tag == "figcaption":
            self._buf = []
            self._stack.append(tag)
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "figure"):
            self._flush_paragraph()
            self._stack.append(tag)
            return

        if tag in _INLINE_TAGS:
            if self._in_cell:
                # collect plain for table cells
                return
            self._push_inline(tag, attrs)
            self._stack.append(tag)
            return

        self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return

        if self._stack and self._stack[-1] == tag:
            self._stack.pop()

        if tag == "li":
            self._emit_list_item()
            return

        if tag in ("ul", "ol"):
            if self._list_stack:
                self._list_stack.pop()
            self.flows.append(Spacer(1, 1.2 * mm))
            return

        if tag == "pre":
            self._in_pre = False
            code = "".join(self._pre_buf)
            self._pre_buf = []
            self.flows.append(Preformatted(code or " ", self.styles["code"]))
            return

        if tag == "blockquote":
            text = "".join(self._buf).strip()
            self._buf = []
            if text:
                text = re.sub(r"\s+", " ", text)
                self.flows.append(Paragraph(text, self.styles["quote"]))
            return

        if tag in ("td", "th"):
            self._in_cell = False
            cell = "".join(self._cell_buf).strip()
            self._cell_buf = []
            self._table_row.append(cell)
            return

        if tag == "tr":
            if self._table_row:
                self._table_rows.append(self._table_row)
            self._table_row = []
            return

        if tag == "table":
            self._in_table = False
            self._finish_table()
            return

        if tag == "figcaption":
            text = "".join(self._buf).strip()
            self._buf = []
            if text:
                self.flows.append(Paragraph(safe(text), self.styles["caption"]))
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush_paragraph(tag)
            return

        if tag in ("p", "div"):
            self._flush_paragraph("body")
            return

        if tag in _INLINE_TAGS:
            self._pop_inline()
            return

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_pre:
            self._pre_buf.append(data)
            return
        if self._in_cell:
            self._cell_buf.append(data)
            return
        # Escape raw text for Paragraph XML
        self._buf.append(safe(data))

    def handle_entityref(self, name: str) -> None:
        # convert_charrefs=True normally handles this; keep fallback
        self.handle_data(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.handle_data(f"&#{name};")


def html_to_flowables(
    html: str,
    styles: dict,
    content_width: float,
) -> Tuple[List[Flowable], List[Tuple[str, str, int]]]:
    """Convert CKEditor HTML into flowables + heading bookmarks.

    Returns ``(flows, [(bookmark_name, title, level), ...])``.
    """
    if not (html or "").strip():
        return [Paragraph("&nbsp;", styles["body"])], []

    cleaned = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    parser = _HtmlToFlowables(styles, content_width)
    try:
        parser.feed(cleaned)
        parser.close()
    except Exception:
        plain = re.sub(r"<[^>]+>", " ", cleaned)
        plain = re.sub(r"\s+", " ", plain).strip()
        return [Paragraph(safe(plain) or "&nbsp;", styles["body"])], []

    flows = parser.flows
    if not flows:
        plain = re.sub(r"<[^>]+>", " ", cleaned)
        plain = re.sub(r"\s+", " ", plain).strip()
        return [Paragraph(safe(plain) or "&nbsp;", styles["body"])], []
    return flows, parser.heading_bookmarks


def extract_headings_from_html(html: str) -> List[Tuple[int, str]]:
    """Return [(level, title), ...] for TOC generation."""
    import html as html_lib

    pattern = re.compile(
        r"<h([1-6])[^>]*>(.*?)</h\1>",
        re.IGNORECASE | re.DOTALL,
    )
    out: List[Tuple[int, str]] = []
    for m in pattern.finditer(html or ""):
        level = int(m.group(1))
        raw = re.sub(r"<[^>]+>", "", m.group(2))
        title = re.sub(r"\s+", " ", html_lib.unescape(raw)).strip()
        if title:
            out.append((level, title))
    return out
