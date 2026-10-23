"""Create slim text-only PDF excerpts via pymupdf extract + reportlab."""
from __future__ import annotations

from pathlib import Path

import fitz
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import inch

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_subsets"
OUT.mkdir(parents=True, exist_ok=True)

JOBS = [
    (
        Path(r"C:\PC\Yoka\Misc\genova-rag\samples\Education\calculus-volume-1_-_WEB.pdf"),
        OUT / "math-calculus-text.pdf",
        25,
    ),
    (
        Path(r"C:\PC\Yoka\Misc\genova-rag\samples\Education\Chemistry-LR.pdf"),
        OUT / "chemistry-text.pdf",
        25,
    ),
    (
        Path(r"C:\PC\Yoka\Misc\genova-rag\samples\Education\upsc-prelims-complete.pdf"),
        OUT / "upsc-text.pdf",
        25,
    ),
]


def extract_text(src: Path, max_pages: int) -> str:
    doc = fitz.open(src)
    n = min(len(doc), max_pages)
    parts: list[str] = []
    for i in range(n):
        t = doc[i].get_text("text") or ""
        t = t.replace("\x00", "").strip()
        if t:
            parts.append(f"--- Page {i + 1} ---\n{t}")
    doc.close()
    return "\n\n".join(parts)


def write_text_pdf(text: str, dest: Path) -> None:
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize = 9
    body.leading = 11
    story = []
    # Escape XML specials for reportlab Paragraph
    for block in text.split("\n"):
        safe = (
            block.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if not safe.strip():
            story.append(Spacer(1, 6))
            continue
        story.append(Paragraph(safe, body))
    doc = SimpleDocTemplate(
        str(dest),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    doc.build(story)


def main() -> None:
    for src, dest, pages in JOBS:
        print(f"Extracting {src.name} first {pages} pages...")
        text = extract_text(src, pages)
        print(f"  chars={len(text)}")
        write_text_pdf(text, dest)
        size = dest.stat().st_size
        print(f"  wrote {dest} ({size:,} bytes)")
        if size > 2_000_000:
            raise SystemExit(f"Too large: {dest} {size}")


if __name__ == "__main__":
    main()
