"""Chat conversation → premium ebook PDF."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, List, Optional

from reportlab.platypus import (
    Flowable,
    HRFlowable,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.units import mm

from app.publishing.builder import (
    BookmarkAnchor,
    auto_subtitle_from_questions,
    build_cover,
    build_toc,
    finalize_pdf,
    question_preview,
)
from app.publishing.design import (
    ACCENT,
    BORDER,
    ASSIST_BG,
    build_styles,
    content_width,
)
from app.publishing.markdown import inline, markdown_to_flowables, safe


def _extract_source_chunks(raw: Optional[str]) -> List[dict]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []

    chunks: Any = parsed
    if isinstance(parsed, dict):
        chunks = parsed.get("chunks") or parsed.get("sources") or []
    if not isinstance(chunks, list):
        return []

    out: List[dict] = []
    for item in chunks:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename") or item.get("original_name") or "Document"
        page = item.get("page")
        text = item.get("text") or item.get("excerpt") or ""
        conf = item.get("confidence", item.get("score"))
        out.append(
            {
                "filename": str(filename),
                "page": page,
                "text": str(text),
                "confidence": conf,
            }
        )
    return out


def _format_confidence(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num <= 1:
        num *= 100
    return f"{num:.0f}%"


def _source_flowables(chunks: List[dict], styles: dict) -> List[Flowable]:
    if not chunks:
        return []
    flows: List[Flowable] = [
        Paragraph(f"References · {len(chunks)} source{'s' if len(chunks) != 1 else ''}", styles["source_title"]),
    ]
    for i, src in enumerate(chunks, 1):
        page = src.get("page")
        page_txt = (
            f"p. {page}"
            if isinstance(page, int) and page >= 1
            else "page n/a"
        )
        conf = _format_confidence(src.get("confidence"))
        meta = page_txt + (f" · confidence {conf}" if conf else "")
        flows.append(Paragraph(f"[{i}] {inline(src['filename'])}", styles["source_item"]))
        flows.append(Paragraph(safe(meta), styles["source_meta"]))
        excerpt = (src.get("text") or "").strip()
        if excerpt:
            clipped = excerpt[:200] + ("…" if len(excerpt) > 200 else "")
            flows.append(Paragraph(inline(clipped), styles["source_excerpt"]))
    return flows


def _role_header(
    label: str,
    timestamp: str,
    *,
    badge_style,
    time_style,
    accent,
    bg,
    width: float,
) -> Table:
    header = Table(
        [
            [
                Paragraph(label, badge_style),
                Paragraph(timestamp, time_style) if timestamp else "",
            ]
        ],
        colWidths=[width * 0.62, width * 0.38],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.45, BORDER),
                ("LINEBEFORE", (0, 0), (0, -1), 2.4, accent),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    return header


def format_chat_pdf(doc, messages: List) -> bytes:
    """Render chat history as a professionally typeset study ebook."""
    styles = build_styles()
    cw = content_width()

    title_name = getattr(doc, "original_name", None) or "Chat session"
    user_msgs = [m for m in messages if getattr(m, "role", "") == "user"]
    questions = [(m.content or "").strip() for m in user_msgs if (m.content or "").strip()]
    subtitle = auto_subtitle_from_questions(questions)
    n_user = len(user_msgs)
    n_assist = sum(1 for m in messages if getattr(m, "role", "") == "assistant")

    story: List[Flowable] = []
    story.extend(
        build_cover(
            styles,
            title=title_name,
            subtitle=subtitle,
            kind_label="Conversation ebook",
            meta_lines=[
                f"<b>Questions</b>  {n_user}",
                f"<b>Answers</b>  {n_assist}",
                f"<b>Total messages</b>  {len(messages)}",
                f"<b>Export date</b>  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ],
        )
    )

    # TOC entries from actual question titles
    toc_entries = []
    for idx, q in enumerate(questions, 1):
        bookmark = f"ch{idx}"
        toc_entries.append((bookmark, f"Chapter {idx} · {question_preview(q, 70)}", 0))
    story.extend(build_toc(styles, toc_entries, heading="Table of contents"))

    q_index = 0
    for msg in messages:
        is_user = getattr(msg, "role", "") == "user"
        timestamp = ""
        if getattr(msg, "created_at", None):
            try:
                timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M")
            except Exception:
                timestamp = ""

        if is_user:
            q_index += 1
            chapter_title = question_preview(msg.content or "", 100) or f"Question {q_index}"
            bookmark = f"ch{q_index}"
            story.append(BookmarkAnchor(bookmark, f"Chapter {q_index}: {chapter_title}", 0))
            story.append(
                Paragraph(f"CHAPTER {q_index}", styles["section_kicker"])
            )
            story.append(Paragraph(inline(chapter_title), styles["chapter_title"]))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.6,
                    color=ACCENT,
                    spaceBefore=0,
                    spaceAfter=3 * mm,
                )
            )
            if timestamp:
                story.append(Paragraph(timestamp, styles["timestamp"]))
                story.append(Spacer(1, 2 * mm))
            # Full question body if longer than title preview
            body_text = (msg.content or "").strip()
            if body_text and body_text != chapter_title:
                story.extend(markdown_to_flowables(body_text, styles, cw))
            story.append(Spacer(1, 3 * mm))
            continue

        # Assistant response — editorial content under the chapter
        label = "Response" if q_index else "Assistant"
        story.append(
            _role_header(
                label,
                timestamp,
                badge_style=styles["badge_assistant"],
                time_style=styles["timestamp"],
                accent=ACCENT,
                bg=ASSIST_BG,
                width=cw,
            )
        )
        story.append(Spacer(1, 2 * mm))
        story.extend(markdown_to_flowables(msg.content or "", styles, cw))

        chunks = _extract_source_chunks(getattr(msg, "sources_json", None))
        if chunks:
            story.append(Spacer(1, 1.5 * mm))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.4,
                    color=BORDER,
                    spaceBefore=0.5 * mm,
                    spaceAfter=1.5 * mm,
                )
            )
            story.extend(_source_flowables(chunks, styles))

        story.append(Spacer(1, 5 * mm))

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=BORDER,
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        )
    )
    story.append(
        Paragraph(
            f"Generated by Genova by Yoka · {datetime.now().strftime('%Y-%m-%d')} · "
            "Re-export anytime from the chat session menu.",
            styles["footer_note"],
        )
    )

    return finalize_pdf(
        story,
        title=f"{title_name} — Genova",
        subject="Genova conversation ebook",
    )
