"""Copy small sample PDFs and build 15-20 page text excerpts for huge books."""
from __future__ import annotations

import shutil
from pathlib import Path

import fitz
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(r"C:\PC\Yoka\PDF-Assistant-RAG-dev\PDF-Assistant-RAG-dev")
SRC = Path(r"C:\PC\Yoka\Misc\genova-rag\samples")
DST = ROOT / "samples"
MAX_FULL_BYTES = 45 * 1024 * 1024
EXCERPT_PAGES = 18

COPY_DIRS = ["Medcare", "payment"]
EXCERPT_JOBS = [
    ("Education/calculus-volume-1_-_WEB.pdf", "Education/calculus-volume-1_excerpt.pdf"),
    ("Education/Chemistry-LR.pdf", "Education/Chemistry-LR_excerpt.pdf"),
    ("Education/upsc-prelims-complete.pdf", "Education/upsc-prelims_excerpt.pdf"),
    ("Education/AlgebraAndTrigonometry-OP_1tE6R5r.pdf", "Education/AlgebraAndTrigonometry_excerpt.pdf"),
]


def extract_text(src: Path, max_pages: int) -> str:
    doc = fitz.open(src)
    n = min(len(doc), max_pages)
    parts: list[str] = []
    for i in range(n):
        t = (doc[i].get_text("text") or "").replace("\x00", "").strip()
        if t:
            parts.append(f"--- Page {i + 1} ---\n{t}")
    doc.close()
    return "\n\n".join(parts)


def write_text_pdf(text: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize = 9
    body.leading = 11
    story = []
    for block in text.split("\n"):
        safe = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if not safe.strip():
            story.append(Spacer(1, 6))
            continue
        story.append(Paragraph(safe, body))
    SimpleDocTemplate(
        str(dest),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    ).build(story)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for sub in COPY_DIRS:
        src_dir = SRC / sub
        dst_dir = DST / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        for pdf in sorted(src_dir.glob("*.pdf")):
            if pdf.stat().st_size > MAX_FULL_BYTES:
                print(f"SKIP large: {pdf}")
                continue
            dest = dst_dir / pdf.name
            shutil.copy2(pdf, dest)
            print(f"COPIED {pdf.name} ({pdf.stat().st_size:,} bytes)")

    for rel_src, rel_dst in EXCERPT_JOBS:
        src = SRC / rel_src
        dest = DST / rel_dst
        if not src.exists():
            print(f"MISSING {src}")
            continue
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"EXISTS {dest.name}")
            continue
        print(f"EXCERPT {src.name} -> {dest.name} ({EXCERPT_PAGES} pages)")
        text = extract_text(src, EXCERPT_PAGES)
        write_text_pdf(text, dest)
        print(f"  wrote {dest.stat().st_size:,} bytes")

    print("\nSAMPLES:")
    for p in sorted(DST.rglob("*.pdf")):
        print(f"  {p.relative_to(DST)} ({p.stat().st_size:,})")


if __name__ == "__main__":
    main()
