# Enterprise Agentic RAG System — Backend
# Register ``import ste`` alias for the in-app STE package (app/ste).
try:
    from app.ste_path import install_ste_alias

    install_ste_alias()
except Exception:
    pass
