"""Expose ``app.ste`` as top-level ``import ste`` (no Docker path hacks).

Production images already copy ``backend/app`` but historically omitted a
sibling ``ste/`` package, which caused ``ModuleNotFoundError: No module named
'ste'`` and DetectDocumentText junk tables. STE now lives at ``app/ste`` so
normal app-code deploys include it. This module registers the ``ste`` alias.
"""
from __future__ import annotations

import importlib
import sys


def install_ste_alias() -> None:
    """Map top-level ``ste`` → ``app.ste``."""
    existing = sys.modules.get("ste")
    if existing is not None and getattr(existing, "__file__", None):
        return
    try:
        app_ste = importlib.import_module("app.ste")
    except ImportError:
        return
    sys.modules["ste"] = app_ste


# Install on import so API, Celery, and ``import app`` callers get ``ste``.
install_ste_alias()
