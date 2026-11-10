"""Minimal Dropbox API v2 client and OAuth helpers.

Only the endpoints needed by the RAG document source are implemented:

* OAuth 2.0 authorize / token exchange / refresh (offline access)
* ``files/list_folder`` (+ ``continue``) for folders and shared links
* ``files/download`` / ``sharing/get_shared_link_file`` for PDF bytes
* ``sharing/get_shared_link_metadata`` and ``users/get_current_account``
* Public ``dl=1`` download fallback for shared links when no API token is
  available (Dropbox serves shared folders as a ZIP archive that way).

The module intentionally uses ``httpx`` directly (already a project
dependency) instead of adding the Dropbox SDK.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

API_BASE = "https://api.dropboxapi.com/2"
CONTENT_BASE = "https://content.dropboxapi.com/2"
OAUTH_AUTHORIZE_URL = "https://www.dropbox.com/oauth2/authorize"
OAUTH_TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"

# Read-only scopes. Shared-link downloads fall back to the public archive when
# the app has not enabled ``sharing.read`` in the App Console, so the connector
# does not require that optional scope for OAuth to succeed.
DROPBOX_SCOPES = (
    "files.metadata.read",
    "files.content.read",
)

ALLOWED_SHARED_LINK_HOSTS = {
    "www.dropbox.com",
    "dropbox.com",
    "dl.dropboxusercontent.com",
}

_USER_AGENT = "GenovaAI-Dropbox/1.0"
_MAX_PUBLIC_DOWNLOAD_BYTES = 300 * 1024 * 1024  # 300 MB safety cap


class DropboxError(Exception):
    """Base class for Dropbox failures."""


class DropboxNotConfigured(DropboxError):
    """App key/secret are missing from the backend environment."""


class DropboxAuthError(DropboxError):
    """Access token expired/revoked or OAuth exchange failed — reconnect."""


class DropboxPermissionDenied(DropboxError):
    """The folder/file is private or the app lacks the required scope."""


class DropboxNotFound(DropboxError):
    """The path/link no longer exists."""


class DropboxNetworkError(DropboxError):
    """Transient transport or server failure."""


class DropboxInvalidLink(DropboxError):
    """The pasted URL is not a Dropbox shared link."""


def is_configured() -> bool:
    return bool(settings.DROPBOX_APP_KEY and settings.DROPBOX_APP_SECRET)


def _require_configured() -> None:
    if not is_configured():
        raise DropboxNotConfigured(
            "Dropbox is not configured on the server (DROPBOX_APP_KEY / "
            "DROPBOX_APP_SECRET are missing)."
        )


# ── Shared links ─────────────────────────────────────────────────────────────


def normalize_shared_link(url: str) -> str:
    """Validate and canonicalise a Dropbox shared link.

    Drops the web-only ``dl``/``raw`` switches (the API ignores them) while
    keeping ``rlkey``/``st`` so the link keeps working for restricted shares.
    """
    raw = (url or "").strip()
    if not raw:
        raise DropboxInvalidLink("Dropbox link is empty")

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise DropboxInvalidLink("Dropbox link must start with http:// or https://")

    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_SHARED_LINK_HOSTS:
        raise DropboxInvalidLink(
            "That is not a Dropbox link. Paste a dropbox.com shared file or folder link."
        )

    if host == "dl.dropboxusercontent.com":
        return raw

    path = parsed.path or ""
    if not (
        path.startswith("/scl/")
        or path.startswith("/sh/")
        or path.startswith("/s/")
    ):
        raise DropboxInvalidLink(
            "That Dropbox link is not a shared link. Use 'Copy link' in Dropbox."
        )

    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in {"dl", "raw"}
    ]
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode(query),
            "",
        )
    )


def shared_link_key(url: str) -> str:
    """Stable identifier for a shared link (used before metadata is known)."""
    normalized = normalize_shared_link(url)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return f"link:{digest}"


def as_direct_download_url(url: str) -> str:
    """Turn a shared link into Dropbox's public ``dl=1`` download URL."""
    parsed = urlparse(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in {"dl", "raw"}
    ]
    query.append(("dl", "1"))
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode(query),
            "",
        )
    )


def _client(*, timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    )


# ── OAuth ────────────────────────────────────────────────────────────────────


def build_authorize_url(state: str) -> str:
    """Build the Dropbox consent URL (offline access → refresh token)."""
    _require_configured()
    params = {
        "client_id": settings.DROPBOX_APP_KEY,
        "response_type": "code",
        "token_access_type": "offline",
        "redirect_uri": settings.DROPBOX_REDIRECT_URI,
        "state": state,
        "scope": " ".join(DROPBOX_SCOPES),
    }
    return f"{OAUTH_AUTHORIZE_URL}?{urlencode(params)}"


def _token_request(data: dict[str, str]) -> dict[str, Any]:
    _require_configured()
    payload = dict(data)
    payload.setdefault("client_id", settings.DROPBOX_APP_KEY)
    payload.setdefault("client_secret", settings.DROPBOX_APP_SECRET)

    try:
        with _client() as client:
            response = client.post(OAUTH_TOKEN_URL, data=payload)
    except httpx.HTTPError as exc:
        raise DropboxNetworkError(f"Network error talking to Dropbox: {exc}") from exc

    if response.status_code in (400, 401, 403):
        raise DropboxAuthError(
            "Dropbox rejected the OAuth credentials. Reconnect Dropbox and try again."
        )
    if response.status_code >= 500:
        raise DropboxNetworkError(
            f"Dropbox token endpoint returned HTTP {response.status_code}"
        )
    if response.status_code != 200:
        raise DropboxAuthError(
            f"Dropbox token request failed with HTTP {response.status_code}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise DropboxNetworkError("Dropbox returned an invalid token response") from exc


def exchange_code(code: str) -> dict[str, Any]:
    """Exchange an authorization code for access + refresh tokens."""
    _require_configured()
    return _token_request(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": settings.DROPBOX_REDIRECT_URI,
        }
    )


def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    """Exchange a stored refresh token for a short-lived access token."""
    return _token_request(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    )


def app_access_token() -> str:
    """App-level token (client_credentials) for shared-link listing/downloads."""
    data = _token_request({"grant_type": "client_credentials"})
    token = data.get("access_token")
    if not token:
        raise DropboxAuthError("Dropbox app authentication returned no access token")
    return token


# ── API error mapping ────────────────────────────────────────────────────────


def _handle_response(response: httpx.Response) -> dict[str, Any]:
    if response.status_code == 200:
        try:
            return response.json()
        except ValueError as exc:
            raise DropboxNetworkError("Dropbox returned invalid JSON") from exc

    summary = ""
    try:
        payload = response.json()
        summary = str(payload.get("error_summary") or "")
    except ValueError:
        summary = (response.text or "")[:300]

    lowered = summary.lower()
    if response.status_code == 401:
        raise DropboxAuthError(
            "Dropbox access expired or was revoked. Reconnect Dropbox in Settings."
        )
    if response.status_code == 409:
        if "not_found" in lowered or "path/not_found" in lowered:
            raise DropboxNotFound("Dropbox path not found")
        if "shared_link_is_directory" in lowered:
            raise DropboxInvalidLink("That shared link points to a folder, not a file")
        if (
            "insufficient_scope" in lowered
            or "no_permission" in lowered
            or "access_denied" in lowered
            or "shared_link_access_denied" in lowered
            or "password" in lowered
        ):
            raise DropboxPermissionDenied(
                "Dropbox denied access. Check the link's sharing permissions "
                "or reconnect Dropbox with read access."
            )
        raise DropboxError(summary or "Dropbox request failed")
    if response.status_code == 429:
        raise DropboxNetworkError("Dropbox rate limit reached — try again shortly")
    if response.status_code >= 500:
        raise DropboxNetworkError(
            f"Dropbox is unavailable (HTTP {response.status_code})"
        )
    raise DropboxError(summary or f"Dropbox request failed (HTTP {response.status_code})")


class DropboxClient:
    """Thin wrapper around a bearer-token authenticated Dropbox session."""

    def __init__(self, access_token: str):
        self.access_token = access_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def _rpc(self, endpoint: str, payload: Optional[dict[str, Any]] = None) -> dict:
        headers = {**self._headers(), "Content-Type": "application/json"}
        try:
            with _client() as client:
                response = client.post(
                    f"{API_BASE}/{endpoint}",
                    headers=headers,
                    content=json.dumps(payload if payload is not None else None),
                )
        except httpx.HTTPError as exc:
            raise DropboxNetworkError(f"Network error talking to Dropbox: {exc}") from exc
        return _handle_response(response)

    def _content(
        self,
        endpoint: str,
        arg: dict[str, Any],
    ) -> tuple[dict[str, Any], bytes]:
        headers = {
            **self._headers(),
            "Dropbox-API-Arg": json.dumps(arg, separators=(",", ":")),
            "Content-Type": "application/octet-stream",
        }
        try:
            with _client(timeout=180.0) as client:
                response = client.post(f"{CONTENT_BASE}/{endpoint}", headers=headers)
        except httpx.HTTPError as exc:
            raise DropboxNetworkError(f"Network error downloading from Dropbox: {exc}") from exc

        if response.status_code != 200:
            _handle_response(response)  # raises the mapped error

        metadata: dict[str, Any] = {}
        raw_meta = response.headers.get("Dropbox-API-Result")
        if raw_meta:
            try:
                metadata = json.loads(raw_meta)
            except ValueError:
                metadata = {}
        data = response.content
        if data[:2] == b"PK":
            # A folder was requested as a file — callers must list it first.
            raise DropboxInvalidLink(
                "Dropbox returned a folder archive instead of a single PDF"
            )
        return metadata, data

    # ── Listing ──

    def list_folder_page(
        self,
        path: str = "",
        *,
        shared_link: Optional[str] = None,
        recursive: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": path or "",
            "include_non_downloadable_files": False,
        }
        if shared_link:
            # Dropbox only supports non-recursive listing through a shared link.
            payload["shared_link"] = {"url": shared_link}
        else:
            payload["recursive"] = bool(recursive)
        return self._rpc("files/list_folder", payload)

    def list_folder(
        self,
        path: str = "",
        *,
        shared_link: Optional[str] = None,
        recursive: bool = False,
    ) -> list[dict[str, Any]]:
        result = self.list_folder_page(
            path, shared_link=shared_link, recursive=recursive
        )
        entries = list(result.get("entries") or [])
        while result.get("has_more"):
            result = self._rpc(
                "files/list_folder/continue",
                {"cursor": result.get("cursor")},
            )
            entries.extend(result.get("entries") or [])
        return entries

    def get_metadata(self, path: str) -> dict[str, Any]:
        return self._rpc("files/get_metadata", {"path": path or ""})

    def get_current_account(self) -> dict[str, Any]:
        return self._rpc("users/get_current_account", None)

    def shared_link_metadata(self, url: str) -> dict[str, Any]:
        return self._rpc("sharing/get_shared_link_metadata", {"url": url})

    # ── Downloads ──

    def download(self, path: str) -> bytes:
        """Download a file from the user's Dropbox by path or ID."""
        _metadata, data = self._content("files/download", {"path": path})
        return data

    def download_shared_link(
        self,
        url: str,
        *,
        path: Optional[str] = None,
    ) -> bytes:
        """Download one file from a shared link (optionally inside a folder)."""
        arg: dict[str, Any] = {"url": url}
        if path:
            arg["path"] = path
        _metadata, data = self._content("sharing/get_shared_link_file", arg)
        return data


def download_public_shared_link(url: str) -> tuple[str, bytes, str]:
    """Download a shared link without any account/API token.

    A shared *folder* is returned by Dropbox as a ZIP archive; a shared *file*
    is returned as the file itself. Returns ``(filename, bytes, content_type)``.
    """
    direct = as_direct_download_url(normalize_shared_link(url))
    try:
        with _client(timeout=300.0) as client:
            with client.stream("GET", direct) as response:
                if response.status_code == 404:
                    raise DropboxNotFound("Dropbox shared link was not found")
                if response.status_code in (401, 403):
                    raise DropboxPermissionDenied(
                        "This Dropbox link is not publicly downloadable"
                    )
                if response.status_code >= 500:
                    raise DropboxNetworkError(
                        f"Dropbox download failed with HTTP {response.status_code}"
                    )
                if response.status_code != 200:
                    raise DropboxPermissionDenied(
                        f"Dropbox download failed with HTTP {response.status_code}"
                    )

                content_type = (response.headers.get("content-type") or "").lower()
                if "text/html" in content_type:
                    raise DropboxPermissionDenied(
                        "Dropbox returned a sign-in page instead of a file. "
                        "The link may be private — connect Dropbox and retry."
                    )

                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > _MAX_PUBLIC_DOWNLOAD_BYTES:
                        raise DropboxError(
                            "Shared Dropbox folder is too large to import (over 300 MB)"
                        )
                    chunks.append(chunk)
                data = b"".join(chunks)

                disposition = response.headers.get("content-disposition") or ""
                filename = _filename_from_disposition(disposition)
                if not filename:
                    filename = urlparse(direct).path.rstrip("/").split("/")[-1] or "dropbox"
                return filename, data, content_type
    except DropboxError:
        raise
    except httpx.HTTPError as exc:
        raise DropboxNetworkError(f"Network error downloading Dropbox link: {exc}") from exc


def _filename_from_disposition(disposition: str) -> str:
    import re
    from urllib.parse import unquote

    if not disposition:
        return ""
    match = re.search(r"filename\*=UTF-8''([^;]+)", disposition, re.IGNORECASE)
    if match:
        return unquote(match.group(1)).strip().strip('"')
    match = re.search(r'filename="?([^";]+)"?', disposition, re.IGNORECASE)
    return match.group(1).strip() if match else ""
