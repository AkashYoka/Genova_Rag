"""Shared PDF publishing design tokens and typography.

Visual direction: warm off-white pages, charcoal body, muted secondary,
restrained Genova teal accent — editorial technical ebook, not a chat dump.
"""
from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas

from app.publishing.fonts import MONO, SANS, SANS_BOLD, SANS_ITALIC, font

# ── Colour tokens ────────────────────────────────────────────────────────────
PAGE_BG = colors.HexColor("#FAF8F5")       # warm off-white
SURFACE = colors.HexColor("#FFFFFF")
INK = colors.HexColor("#1A1A1C")           # dark charcoal
MUTED = colors.HexColor("#5C5C66")
FAINT = colors.HexColor("#8A8A94")
BORDER = colors.HexColor("#E2DED8")
RULE = colors.HexColor("#D4CFC7")
ACCENT = colors.HexColor("#0F7C86")        # Genova teal
ACCENT_SOFT = colors.HexColor("#E8F4F5")
CODE_BG = colors.HexColor("#F3F1ED")
TABLE_HEADER = colors.HexColor("#EEF6F7")
TABLE_ALT = colors.HexColor("#F7F5F1")
TABLE_GRID = colors.HexColor("#D8D4CE")
USER_BG = colors.HexColor("#F3F1ED")
ASSIST_BG = colors.HexColor("#FAF9F7")
QUOTE_BG = colors.HexColor("#F5F3EF")
COVER_BAND = colors.HexColor("#0F7C86")

PAGE_SIZE = A4
MARGIN_L = 18 * mm
MARGIN_R = 18 * mm
MARGIN_T = 20 * mm
MARGIN_B = 18 * mm


def content_width() -> float:
    return PAGE_SIZE[0] - MARGIN_L - MARGIN_R


def build_styles() -> dict:
    base = getSampleStyleSheet()
    f_sans = font(SANS)
    f_bold = font(SANS_BOLD)
    f_italic = font(SANS_ITALIC)
    f_mono = font(MONO)
    return {
        "cover_brand": ParagraphStyle(
            "PubCoverBrand",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=10,
            leading=13,
            textColor=ACCENT,
            spaceAfter=8 * mm,
            tracking=1.2,
        ),
        "cover_title": ParagraphStyle(
            "PubCoverTitle",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=26,
            leading=32,
            textColor=INK,
            spaceAfter=4 * mm,
        ),
        "cover_subtitle": ParagraphStyle(
            "PubCoverSubtitle",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=12,
            leading=17,
            textColor=MUTED,
            spaceAfter=8 * mm,
        ),
        "cover_meta": ParagraphStyle(
            "PubCoverMeta",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=9.5,
            leading=14,
            textColor=MUTED,
            spaceAfter=1.5 * mm,
        ),
        "cover_footer": ParagraphStyle(
            "PubCoverFooter",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=8.5,
            leading=11,
            textColor=FAINT,
        ),
        "toc_heading": ParagraphStyle(
            "PubTocHeading",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=16,
            leading=20,
            textColor=INK,
            spaceAfter=5 * mm,
        ),
        "toc_item": ParagraphStyle(
            "PubTocItem",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=10,
            leading=15,
            textColor=INK,
            spaceAfter=2.2 * mm,
        ),
        "toc_item_sub": ParagraphStyle(
            "PubTocItemSub",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=9,
            leading=13,
            textColor=MUTED,
            leftIndent=4 * mm,
            spaceAfter=1.5 * mm,
        ),
        "chapter_title": ParagraphStyle(
            "PubChapterTitle",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=14,
            leading=19,
            textColor=INK,
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        ),
        "section_kicker": ParagraphStyle(
            "PubSectionKicker",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=8,
            leading=10,
            textColor=ACCENT,
            spaceAfter=1 * mm,
        ),
        "badge_user": ParagraphStyle(
            "PubBadgeUser",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=8.5,
            leading=11,
            textColor=INK,
        ),
        "badge_assistant": ParagraphStyle(
            "PubBadgeAssistant",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=8.5,
            leading=11,
            textColor=ACCENT,
        ),
        "timestamp": ParagraphStyle(
            "PubTimestamp",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=7.5,
            leading=10,
            textColor=FAINT,
        ),
        "body": ParagraphStyle(
            "PubBody",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=10,
            leading=15,
            textColor=INK,
            spaceAfter=2.2 * mm,
            alignment=0,
        ),
        "h1": ParagraphStyle(
            "PubH1",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=16,
            leading=21,
            textColor=INK,
            spaceBefore=5 * mm,
            spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "PubH2",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=13,
            leading=17,
            textColor=INK,
            spaceBefore=4 * mm,
            spaceAfter=2.5 * mm,
        ),
        "h3": ParagraphStyle(
            "PubH3",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=11.5,
            leading=15,
            textColor=INK,
            spaceBefore=3.5 * mm,
            spaceAfter=2 * mm,
        ),
        "h4": ParagraphStyle(
            "PubH4",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=10.5,
            leading=14,
            textColor=INK,
            spaceBefore=3 * mm,
            spaceAfter=1.5 * mm,
        ),
        "h5": ParagraphStyle(
            "PubH5",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=10,
            leading=13,
            textColor=MUTED,
            spaceBefore=2.5 * mm,
            spaceAfter=1.5 * mm,
        ),
        "h6": ParagraphStyle(
            "PubH6",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=9.5,
            leading=12,
            textColor=MUTED,
            spaceBefore=2 * mm,
            spaceAfter=1.5 * mm,
        ),
        "list": ParagraphStyle(
            "PubList",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=10,
            leading=14.5,
            textColor=INK,
            leftIndent=5 * mm,
            spaceAfter=1.2 * mm,
        ),
        "list_nested": ParagraphStyle(
            "PubListNested",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=9.5,
            leading=13.5,
            textColor=INK,
            leftIndent=10 * mm,
            spaceAfter=1 * mm,
        ),
        "quote": ParagraphStyle(
            "PubQuote",
            parent=base["Normal"],
            fontName=f_italic,
            fontSize=10,
            leading=15,
            textColor=MUTED,
            leftIndent=4 * mm,
            rightIndent=2 * mm,
            spaceBefore=2 * mm,
            spaceAfter=2.5 * mm,
            backColor=QUOTE_BG,
            borderPadding=4,
        ),
        "code": ParagraphStyle(
            "PubCode",
            parent=base["Code"],
            fontName=f_mono,
            fontSize=8,
            leading=11,
            textColor=INK,
            backColor=CODE_BG,
            leftIndent=2 * mm,
            rightIndent=2 * mm,
            spaceBefore=2 * mm,
            spaceAfter=2.5 * mm,
        ),
        "inline_code": ParagraphStyle(
            "PubInlineCode",
            parent=base["Normal"],
            fontName=f_mono,
            fontSize=8.5,
            leading=12,
            textColor=INK,
        ),
        "caption": ParagraphStyle(
            "PubCaption",
            parent=base["Normal"],
            fontName=f_italic,
            fontSize=8.5,
            leading=11,
            textColor=MUTED,
            alignment=1,
            spaceBefore=1 * mm,
            spaceAfter=3 * mm,
        ),
        "table_cell": ParagraphStyle(
            "PubTableCell",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=8,
            leading=11,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "PubTableHeader",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=8,
            leading=11,
            textColor=INK,
        ),
        "source_title": ParagraphStyle(
            "PubSourceTitle",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=9,
            leading=12,
            textColor=ACCENT,
            spaceBefore=1.5 * mm,
            spaceAfter=2 * mm,
        ),
        "source_item": ParagraphStyle(
            "PubSourceItem",
            parent=base["Normal"],
            fontName=f_bold,
            fontSize=8.5,
            leading=11,
            textColor=INK,
        ),
        "source_meta": ParagraphStyle(
            "PubSourceMeta",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            spaceAfter=0.4 * mm,
        ),
        "source_excerpt": ParagraphStyle(
            "PubSourceExcerpt",
            parent=base["Normal"],
            fontName=f_italic,
            fontSize=7.5,
            leading=10.5,
            textColor=MUTED,
            spaceAfter=2 * mm,
        ),
        "link": ParagraphStyle(
            "PubLink",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=10,
            leading=15,
            textColor=ACCENT,
        ),
        "footer_note": ParagraphStyle(
            "PubFooterNote",
            parent=base["Normal"],
            fontName=f_sans,
            fontSize=8,
            leading=11,
            textColor=FAINT,
        ),
    }


def draw_page_chrome(
    canvas_obj: pdf_canvas.Canvas,
    doc,
    *,
    running_title: str = "Genova",
    show_header: bool = True,
) -> None:
    """Warm page fill + editorial header/footer with page numbers."""
    canvas_obj.saveState()
    page_w, page_h = PAGE_SIZE

    # Warm off-white background (independent of app theme)
    canvas_obj.setFillColor(PAGE_BG)
    canvas_obj.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    # Skip chrome on cover page (page 1 of first pass; multiBuild may vary)
    is_cover = getattr(doc, "_is_cover_page", False) or doc.page == 1 and getattr(
        doc, "_cover_mode", False
    )

    if not is_cover and show_header:
        # Header rule
        canvas_obj.setStrokeColor(RULE)
        canvas_obj.setLineWidth(0.5)
        header_y = page_h - 12 * mm
        canvas_obj.line(MARGIN_L, header_y, page_w - MARGIN_R, header_y)
        canvas_obj.setFont(font(SANS), 7.5)
        canvas_obj.setFillColor(FAINT)
        title = (running_title or "Genova")[:72]
        canvas_obj.drawString(MARGIN_L, header_y + 2.5 * mm, title)
        canvas_obj.drawRightString(
            page_w - MARGIN_R, header_y + 2.5 * mm, "Genova by Yoka"
        )

        # Footer
        footer_y = 10 * mm
        canvas_obj.setStrokeColor(RULE)
        canvas_obj.line(MARGIN_L, footer_y + 4 * mm, page_w - MARGIN_R, footer_y + 4 * mm)
        canvas_obj.setFont(font(SANS), 8)
        canvas_obj.setFillColor(FAINT)
        canvas_obj.drawString(MARGIN_L, footer_y, "Genova · Technical publication")
        canvas_obj.drawRightString(page_w - MARGIN_R, footer_y, f"{doc.page}")

    try:
        canvas_obj.setViewerPreference("PageMode", "UseOutlines")
    except Exception:
        pass
    canvas_obj.restoreState()


def draw_cover_page(canvas_obj: pdf_canvas.Canvas, doc) -> None:
    """Full-bleed cover with teal accent band — used only for page 1."""
    canvas_obj.saveState()
    page_w, page_h = PAGE_SIZE
    canvas_obj.setFillColor(PAGE_BG)
    canvas_obj.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    # Top accent band
    canvas_obj.setFillColor(COVER_BAND)
    canvas_obj.rect(0, page_h - 8 * mm, page_w, 8 * mm, fill=1, stroke=0)

    # Subtle bottom rule
    canvas_obj.setStrokeColor(ACCENT)
    canvas_obj.setLineWidth(1.2)
    canvas_obj.line(MARGIN_L, 28 * mm, page_w - MARGIN_R, 28 * mm)

    canvas_obj.setFont(font(SANS), 8)
    canvas_obj.setFillColor(FAINT)
    canvas_obj.drawString(MARGIN_L, 18 * mm, "Genova by Yoka")
    canvas_obj.drawRightString(page_w - MARGIN_R, 18 * mm, "Confidential · Personal export")
    canvas_obj.restoreState()
