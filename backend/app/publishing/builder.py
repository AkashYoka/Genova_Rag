"""Shared PDF document builder — cover, TOC, bookmarks, page templates."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Callable, List, Optional, Sequence

from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.units import mm

from app.publishing.design import (
    ACCENT,
    BORDER,
    MARGIN_B,
    MARGIN_L,
    MARGIN_R,
    MARGIN_T,
    PAGE_SIZE,
    build_styles,
    content_width,
    draw_cover_page,
    draw_page_chrome,
)
from app.publishing.markdown import inline, safe


class BookmarkAnchor(Flowable):
    """Zero-height marker used to attach a PDF outline entry."""

    def __init__(self, name: str, title: str, level: int = 0):
        super().__init__()
        self._bookmarkName = name
        self._bookmarkTitle = title
        self._bookmarkLevel = level
        self.width = 0
        self.height = 0

    def draw(self) -> None:
        return


class PublishingDoc(BaseDocTemplate):
    def __init__(self, buffer, *, title: str, author: str = "Genova by Yoka", subject: str = ""):
        super().__init__(
            buffer,
            pagesize=PAGE_SIZE,
            title=title,
            author=author,
            subject=subject or title,
            creator="Genova PDF Publishing",
        )
        self._running_title = title[:80]
        self._cover_mode = True

        page_w, page_h = PAGE_SIZE
        frame = Frame(
            MARGIN_L,
            MARGIN_B,
            page_w - MARGIN_L - MARGIN_R,
            page_h - MARGIN_T - MARGIN_B,
            id="body",
            showBoundary=0,
        )

        def _on_page(canvas_obj, doc):
            if doc.page == 1:
                draw_cover_page(canvas_obj, doc)
            else:
                draw_page_chrome(
                    canvas_obj,
                    doc,
                    running_title=self._running_title,
                    show_header=True,
                )

        self.addPageTemplates([PageTemplate(id="pub", frames=[frame], onPage=_on_page)])

    def afterFlowable(self, flowable: Flowable) -> None:
        name = getattr(flowable, "_bookmarkName", None)
        if not name:
            return
        title = getattr(flowable, "_bookmarkTitle", name)
        level = int(getattr(flowable, "_bookmarkLevel", 0) or 0)
        self.canv.bookmarkPage(name)
        try:
            self.canv.addOutlineEntry(title, name, level=level, closed=0)
        except Exception:
            pass


def build_cover(
    styles: dict,
    *,
    brand: str = "GENOVA BY YOKA",
    title: str,
    subtitle: str = "",
    meta_lines: Optional[Sequence[str]] = None,
    kind_label: str = "Technical publication",
) -> List[Flowable]:
    """Editorial cover page content (page chrome drawn separately)."""
    story: List[Flowable] = []
    story.append(Spacer(1, 28 * mm))
    story.append(Paragraph(safe(brand), styles["cover_brand"]))
    story.append(
        HRFlowable(
            width="40%",
            thickness=2.2,
            color=ACCENT,
            spaceBefore=0,
            spaceAfter=6 * mm,
            hAlign="LEFT",
        )
    )
    story.append(Paragraph(safe(kind_label).upper(), styles["section_kicker"]))
    story.append(Paragraph(safe(title), styles["cover_title"]))
    if subtitle:
        story.append(Paragraph(safe(subtitle), styles["cover_subtitle"]))
    else:
        story.append(Spacer(1, 4 * mm))

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=BORDER,
            spaceBefore=2 * mm,
            spaceAfter=5 * mm,
        )
    )

    for line in meta_lines or []:
        story.append(Paragraph(line, styles["cover_meta"]))

    story.append(Spacer(1, 18 * mm))
    story.append(
        Paragraph(
            f"Exported {datetime.now().strftime('%d %B %Y, %H:%M')}",
            styles["cover_footer"],
        )
    )
    story.append(PageBreak())
    return story


def build_toc(
    styles: dict,
    entries: Sequence[tuple],
    *,
    heading: str = "Contents",
) -> List[Flowable]:
    """Clickable table of contents. entries: (bookmark_name, title, level?)."""
    if not entries:
        return []
    story: List[Flowable] = []
    story.append(Paragraph(heading, styles["toc_heading"]))
    story.append(
        HRFlowable(
            width="100%",
            thickness=0.6,
            color=ACCENT,
            spaceBefore=0,
            spaceAfter=4 * mm,
        )
    )
    for entry in entries:
        if len(entry) == 2:
            name, title = entry
            level = 0
        else:
            name, title, level = entry[0], entry[1], entry[2]
        preview = title if len(title) <= 90 else title[:89] + "…"
        style = styles["toc_item"] if level == 0 else styles["toc_item_sub"]
        story.append(
            Paragraph(
                f'<link href="#{safe(name)}"><font color="#1A1A1C">{inline(preview)}</font></link>',
                style,
            )
        )
    story.append(Spacer(1, 3 * mm))
    story.append(PageBreak())
    return story


def finalize_pdf(
    story: List[Flowable],
    *,
    title: str,
    subject: str = "",
) -> bytes:
    buffer = BytesIO()
    doc = PublishingDoc(buffer, title=title, subject=subject)
    doc.build(story)
    return buffer.getvalue()


def auto_subtitle_from_questions(questions: Sequence[str], limit: int = 120) -> str:
    if not questions:
        return "A structured study ebook generated from your Genova conversation."
    first = re_sub_ws(questions[0])
    if len(questions) == 1:
        return first if len(first) <= limit else first[: limit - 1] + "…"
    return (
        f"Covering {len(questions)} topics, beginning with: "
        + (first if len(first) <= 80 else first[:79] + "…")
    )


def re_sub_ws(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", (text or "").strip())


def question_preview(text: str, limit: int = 72) -> str:
    clean = re_sub_ws(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"
