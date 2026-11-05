"""Public Google Drive folder access without user OAuth.

Uses Drive's public embedded folder view (Anyone with the link → Viewer)
to list entries and ``uc?export=download`` to fetch PDFs. Does not bypass
permissions — private folders return an empty/login page and raise
``DrivePermissionDenied``.
"""
from __future__ import annotations

import logging
import re
from html import unescape
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

logger = logging.getLogger(__name__)

PDF_MIME_TYPE = "application/pdf"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class DriveAccessError(Exception):
    """Base class for Drive access failures."""


class DrivePermissionDenied(DriveAccessError):
    """Folder is private or restricted — OAuth required."""


class DriveNotFound(DriveAccessError):
    """Folder ID does not exist or was removed."""


class DriveNetworkError(DriveAccessError):
    """Transient network / transport failure."""


class DriveInvalidFolder(DriveAccessError):
    """URL/ID is not a Drive folder."""


_ENTRY_RE = re.compile(
    r'<div class="flip-entry"[^>]*\bid="entry-([a-zA-Z0-9_-]+)"[^>]*>'
    r'(?P<body>.*?)</div>\s*<div class="flip-entry-last-modified">',
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(
    r'<div class="flip-entry-title">([^<]+)</div>',
    re.IGNORECASE,
)
_HREF_FILE_RE = re.compile(
    r'href="https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)/',
    re.IGNORECASE,
)
_HREF_FOLDER_RE = re.compile(
    r'href="https://drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)',
    re.IGNORECASE,
)
_IMG_TYPE_RE = re.compile(
    r'type/(application/pdf|[^"\']+)"',
    re.IGNORECASE,
)
_ALT_PDF_RE = re.compile(r'alt="PDF"', re.IGNORECASE)


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    )


def fetch_public_folder_page(folder_id: str) -> str:
    """Fetch the public embedded folder HTML (raises typed errors)."""
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
    try:
        with _client() as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        raise DriveNetworkError(f"Network error fetching Drive folder: {exc}") from exc

    if response.status_code == 404:
        raise DriveNotFound("Google Drive folder not found")
    if response.status_code >= 500:
        raise DriveNetworkError(
            f"Drive returned HTTP {response.status_code} while loading the folder"
        )

    text = response.text or ""
    lower = text.lower()
    # Private / sign-in gated pages do not include flip-entries.
    if "flip-entries" not in lower:
        if "accounts.google.com" in lower or "sign in" in lower or "service login" in lower:
            raise DrivePermissionDenied(
                "This Drive folder is private. Connect Google Drive to access it."
            )
        if "sorry" in lower or "not found" in lower:
            raise DriveNotFound("Google Drive folder not found or unavailable")
        raise DrivePermissionDenied(
            "Could not list this Drive folder publicly. "
            "It may be private — connect Google Drive and retry."
        )
    return text


def parse_public_folder_entries(html: str) -> list[dict[str, Any]]:
    """Parse embedded folder HTML into file/folder entries."""
    entries: list[dict[str, Any]] = []
    for match in _ENTRY_RE.finditer(html):
        entry_id = match.group(1)
        body = match.group("body")
        title_m = _TITLE_RE.search(body)
        name = unescape(title_m.group(1)).strip() if title_m else entry_id

        folder_m = _HREF_FOLDER_RE.search(body)
        file_m = _HREF_FILE_RE.search(body)
        is_pdf = bool(_ALT_PDF_RE.search(body)) or bool(
            _IMG_TYPE_RE.search(body)
            and "pdf" in (_IMG_TYPE_RE.search(body).group(1) or "").lower()
        )
        if name.lower().endswith(".pdf"):
            is_pdf = True

        if folder_m and not file_m:
            entries.append(
                {
                    "id": folder_m.group(1),
                    "name": name,
                    "mimeType": FOLDER_MIME_TYPE,
                }
            )
        else:
            file_id = file_m.group(1) if file_m else entry_id
            mime = PDF_MIME_TYPE if is_pdf else "application/octet-stream"
            entries.append(
                {
                    "id": file_id,
                    "name": name,
                    "mimeType": mime,
                }
            )
    return entries


def list_public_pdfs_recursive(folder_id: str, *, max_folders: int = 50) -> list[dict]:
    """BFS public folder tree; return PDF file metadata dicts."""
    pdfs: list[dict] = []
    queue = [folder_id]
    seen: set[str] = set()

    while queue and len(seen) < max_folders:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        html = fetch_public_folder_page(current)
        for item in parse_public_folder_entries(html):
            mime = item.get("mimeType")
            if mime == FOLDER_MIME_TYPE:
                queue.append(item["id"])
            elif mime == PDF_MIME_TYPE or str(item.get("name", "")).lower().endswith(
                ".pdf"
            ):
                row = dict(item)
                row["mimeType"] = PDF_MIME_TYPE
                row["parent_folder_id"] = current
                row["modifiedTime"] = ""  # not available from embedded view
                pdfs.append(row)
    return pdfs


def fetch_public_folder_metadata(folder_id: str) -> dict:
    """Return minimal metadata for a publicly listable folder."""
    html = fetch_public_folder_page(folder_id)
    title_m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    name = unescape(title_m.group(1)).strip() if title_m else folder_id
    if name.lower() in {"google drive", "drive", "yoka"}:
        name = folder_id
    # Confirm at least the flip-entries shell exists (already checked).
    return {
        "id": folder_id,
        "name": name,
        "mimeType": FOLDER_MIME_TYPE,
        "access": "public",
    }


def _extract_confirm_token(html: str) -> str | None:
    # Large-file interstitial: confirm=TOKEN in form/action or download link
    for pattern in (
        r"confirm=([0-9A-Za-z_-]+)",
        r'name="confirm"\s+value="([^"]+)"',
    ):
        m = re.search(pattern, html)
        if m:
            return m.group(1)
    return None


def download_public_file(file_id: str, destination: str) -> None:
    """Download a publicly shared Drive file to ``destination``."""
    base = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        with _client() as client:
            response = client.get(base)
            content_type = (response.headers.get("content-type") or "").lower()
            # Virus-scan warning HTML for large files
            if "text/html" in content_type or response.content[:4] != b"%PDF":
                token = _extract_confirm_token(response.text or "")
                if token:
                    response = client.get(
                        f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
                    )
                elif "text/html" in content_type:
                    # Might still be PDF without magic if weird; try cookies from first hit
                    for cookie_name, cookie_value in response.cookies.items():
                        if cookie_name.startswith("download_warning"):
                            response = client.get(
                                f"https://drive.google.com/uc?export=download&id={file_id}&confirm={cookie_value}"
                            )
                            break

            if response.status_code in (401, 403):
                raise DrivePermissionDenied(
                    "This Drive file is not publicly downloadable"
                )
            if response.status_code == 404:
                raise DriveNotFound("Drive file not found")
            if response.status_code >= 500:
                raise DriveNetworkError(
                    f"Drive download failed with HTTP {response.status_code}"
                )

            data = response.content
            if not data:
                raise DriveNetworkError("Empty response downloading Drive file")
            # Reject obvious non-PDF HTML error pages
            if data[:4] != b"%PDF" and b"<html" in data[:200].lower():
                raise DrivePermissionDenied(
                    "Drive returned an HTML page instead of a PDF "
                    "(file may be private or require sign-in)"
                )

            with open(destination, "wb") as handle:
                handle.write(data)
    except DriveAccessError:
        raise
    except httpx.HTTPError as exc:
        raise DriveNetworkError(f"Network error downloading Drive file: {exc}") from exc


def probe_public_folder(folder_id: str) -> tuple[dict, list[dict]]:
    """
    Validate public access and return (metadata, pdf_list).

    Raises DrivePermissionDenied / DriveNotFound / DriveNetworkError.
    """
    meta = fetch_public_folder_metadata(folder_id)
    pdfs = list_public_pdfs_recursive(folder_id)
    # Prefer a friendlier name from the first PDF parent page title if meta is an id
    return meta, pdfs
