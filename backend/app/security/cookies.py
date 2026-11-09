"""HttpOnly auth cookie helpers (static-export compatible — no Next BFF)."""
from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Response

from app.config import get_settings

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"
CSRF_COOKIE = "csrf_token"


def _cookie_common() -> dict:
    settings = get_settings()
    secure = bool(settings.COOKIE_SECURE) or settings.ENVIRONMENT == "production"
    samesite = (settings.COOKIE_SAMESITE or "lax").lower()
    if samesite not in {"lax", "strict", "none"}:
        samesite = "lax"
    # SameSite=None requires Secure
    if samesite == "none":
        secure = True
    kwargs: dict = {
        "httponly": True,
        "secure": secure,
        "samesite": samesite,
        "path": "/",
    }
    domain = (settings.COOKIE_DOMAIN or "").strip()
    if domain:
        kwargs["domain"] = domain
    return kwargs


def set_auth_cookies(
    response: Response,
    *,
    access_token: str,
    refresh_token: str,
    csrf_token: Optional[str] = None,
) -> str:
    """Attach auth cookies. Returns the CSRF token (also set as non-HttpOnly cookie)."""
    settings = get_settings()
    common = _cookie_common()
    access_max = int(settings.JWT_ACCESS_EXPIRY_MINUTES) * 60
    refresh_max = int(settings.JWT_REFRESH_EXPIRY_DAYS) * 24 * 3600

    response.set_cookie(
        key=ACCESS_COOKIE,
        value=access_token,
        max_age=access_max,
        **common,
    )
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=refresh_token,
        max_age=refresh_max,
        **common,
    )

    csrf = csrf_token or secrets.token_urlsafe(32)
    # Double-submit CSRF cookie must be readable by JS.
    csrf_kwargs = dict(common)
    csrf_kwargs["httponly"] = False
    response.set_cookie(
        key=CSRF_COOKIE,
        value=csrf,
        max_age=refresh_max,
        **csrf_kwargs,
    )
    return csrf


def clear_auth_cookies(response: Response) -> None:
    common = _cookie_common()
    for key in (ACCESS_COOKIE, REFRESH_COOKIE, CSRF_COOKIE):
        # Match attributes used when setting so browsers clear reliably.
        response.delete_cookie(
            key=key,
            path=common.get("path", "/"),
            domain=common.get("domain"),
            secure=common.get("secure", False),
            httponly=(key != CSRF_COOKIE),
            samesite=common.get("samesite", "lax"),
        )
