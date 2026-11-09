"""Double-submit CSRF protection for cookie-authenticated browser sessions."""
from __future__ import annotations

import hmac
import secrets
from typing import Optional

from fastapi import HTTPException, Request, status

from app.security.cookies import ACCESS_COOKIE, CSRF_COOKIE, REFRESH_COOKIE

CSRF_HEADER = "X-CSRF-Token"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


def issue_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def ensure_csrf(request: Request) -> None:
    """Require CSRF header when the request authenticates via cookies only.

    Bearer / API-key Authorization bypasses CSRF (non-browser or explicit clients).
    Public unauthenticated POSTs (login/register) have no session cookies yet.
    """
    if request.method.upper() in _SAFE_METHODS:
        return

    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        # Explicit bearer (incl. API keys) — not cookie-session CSRF surface.
        return

    has_session_cookie = bool(
        request.cookies.get(ACCESS_COOKIE) or request.cookies.get(REFRESH_COOKIE)
    )
    if not has_session_cookie:
        return

    cookie_token = request.cookies.get(CSRF_COOKIE)
    header_token = request.headers.get(CSRF_HEADER)
    if not cookie_token or not header_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing",
        )
    if not hmac.compare_digest(cookie_token, header_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token mismatch",
        )


def read_csrf_cookie(request: Request) -> Optional[str]:
    return request.cookies.get(CSRF_COOKIE)
