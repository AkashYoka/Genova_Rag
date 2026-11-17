"""In-repo STE (Textract AnalyzeDocument edition) for Genova PDF table extraction.

Primary path: render PDF pages → AWS Textract ``AnalyzeDocument`` TABLES →
optional Bedrock Nova reconstruction → fully cellular validation/export.

Import as ``ste`` (``backend/`` is on ``sys.path``). Wired by
``app.rag.ingestion.pdf_layout.ste_tables`` behind ``PDF_STE_TABLES_ENABLED``.
"""

__version__ = "0.3.0"
