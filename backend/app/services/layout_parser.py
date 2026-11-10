"""Central layout-aware PDF orchestration.

The heavy lifting lives in ``app.rag.ingestion.pdf_layout``. This module keeps
the historical ``AdvancedPDFParser`` API while routing PDF ingestion through
the classifier → OCR/layout → structure → semantic-chunk pipeline.

Legacy text extraction remains available in
``app.rag.chunking.chunker.extract_pdf`` for callers that need page records
rather than chunks.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pymupdf as fitz

from app.rag.ingestion.pdf_layout import chunk_pdf, parse_pdf
from app.rag.ingestion.pdf_layout.types import ParsedDocument

logger = logging.getLogger(__name__)


class AdvancedPDFParser:
    """Layout-aware parser façade used by the ingestion pipeline."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self._parsed: Optional[ParsedDocument] = None

    def parse(
        self,
        *,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> ParsedDocument:
        if self._parsed is None:
            self._parsed = parse_pdf(
                self.pdf_path,
                document_id=document_id,
                document_name=document_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        return self._parsed

    def chunk_document(
        self,
        *,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self.parse(
            document_id=document_id,
            document_name=document_name or os.path.basename(self.pdf_path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ).chunks

    def extract_structured_text(self) -> List[Dict[str, Any]]:
        """Return page-level records for legacy callers.

        The ``text`` field is the reconstructed layout text (headings,
        paragraphs, lists, tables, captions), not a flat OCR dump.
        """
        parsed = self.parse(document_name=os.path.basename(self.pdf_path))
        pages: List[Dict[str, Any]] = []
        for page in parsed.pages:
            text = "\n\n".join(element.text for element in page.elements if element.text.strip())
            pages.append(
                {
                    "page_number": page.page_number,
                    "text": text,
                    "type": "ocr_layout" if page.strategy == "ocr" else "text_layout",
                    "ocr": page.strategy == "ocr",
                    "tables": len(page.tables),
                    "status": page.status,
                }
            )
        return pages

    def process_embedded_images(self, page_num: int, page_obj: fitz.Page) -> List[str]:
        """Optional visual summaries for embedded (non-page-sized) images.

        Fully scanned pages are skipped: their text is already recovered by OCR,
        and sending a full-page scan to a vision model adds cost without adding
        evidence. The method remains for compatibility with the previous API.
        """
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return []
        try:
            from google import genai
        except Exception as exc:
            logger.debug("Gemini client unavailable; skipping vision summaries: %s", exc)
            return []

        try:
            infos = page_obj.get_image_info()
            page_area = max(1.0, float(page_obj.rect.width) * float(page_obj.rect.height))
            if any(
                (float(info.get("bbox", (0, 0, 0, 0))[2]) - float(info.get("bbox", (0, 0, 0, 0))[0]))
                * (float(info.get("bbox", (0, 0, 0, 0))[3]) - float(info.get("bbox", (0, 0, 0, 0))[1]))
                >= page_area * 0.8
                for info in infos
            ):
                return []
            client = genai.Client(api_key=api_key)
        except Exception as exc:
            logger.warning("Vision processing skipped for page %s: %s", page_num + 1, exc)
            return []

        descriptions: List[str] = []
        for image in page_obj.get_images(full=True):
            try:
                base_image = self.doc.extract_image(image[0])
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        genai.types.Part.from_bytes(
                            data=base_image["image"], mime_type="image/jpeg"
                        ),
                        "Describe the numbers, structure, trends, and labels in this "
                        "document figure so it can be retrieved as text.",
                    ],
                )
                if response.text:
                    descriptions.append(response.text)
            except Exception as exc:
                logger.debug("Vision summary skipped for page %s: %s", page_num + 1, exc)
        return descriptions

    @property
    def doc(self) -> fitz.Document:
        if not hasattr(self, "_doc"):
            self._doc = fitz.open(self.pdf_path)
        return self._doc

    def ingest_document(self) -> List[Dict[str, Any]]:
        """Return structured page records plus optional image summaries."""
        payload = self.extract_structured_text()
        try:
            for page_num in range(len(self.doc)):
                summaries = self.process_embedded_images(page_num, self.doc.load_page(page_num))
                for summary in summaries:
                    payload.append(
                        {
                            "page_number": page_num + 1,
                            "text": f"[Visual Data Extraction Summary]: {summary}",
                            "type": "visual_image_summary",
                        }
                    )
        finally:
            if hasattr(self, "_doc"):
                self._doc.close()
                del self._doc
        return payload

    def __del__(self):
        if hasattr(self, "_doc"):
            try:
                self._doc.close()
            except Exception:
                pass
