"""In-repo STE (Textract AnalyzeDocument edition) for Genova PDF table extraction.

Primary path: render PDF pages → AWS Textract ``AnalyzeDocument`` TABLES →
optional Bedrock Nova reconstruction → fully cellular validation/export.

Lives at ``app/ste`` (shipped with the app package). Import as ``app.ste`` or
top-level ``ste`` via ``app.ste_path.install_ste_alias``. Wired by
``app.rag.ingestion.pdf_layout.ste_tables`` behind ``PDF_STE_TABLES_ENABLED``.
"""


__version__ = "0.3.0"
