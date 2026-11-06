"""Per-user AWS S3 storage for uploaded documents and CKEditor sticky notes.

Layout
------
    s3://<bucket>/NoteBase/{user_id}/pdfs/<stored-filename>
    s3://<bucket>/NoteBase/{user_id}/sticky_notes/notes.json

``S3_BASE_PREFIX`` defaults to ``NoteBase``. The ``user_id`` segment always
comes from the authenticated request, so a user can only ever address their
own prefix. Credentials are resolved by the standard AWS provider chain
(``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_REGION``, shared
credentials file, or IAM role); nothing is hardcoded or committed.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

NOTES_FILENAME = "notes.json"
_NOTES_CONTENT_TYPE = "application/json; charset=utf-8"
_FOLDER_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._\-]+$")

_client_lock = threading.Lock()
_cached_client = None
_cached_bucket: Optional[str] = None
_config_logged = False


class S3StorageError(RuntimeError):
    """Raised when S3 is configured but an operation could not be completed."""


# ── Configuration helpers ─────────────────────────────────────────────────────

def is_configured() -> bool:
    """True when an S3 bucket is configured for per-user storage."""
    return bool((settings.S3_BUCKET or settings.AWS_S3_BUCKET or "").strip())


def _bucket() -> str:
    bucket = (settings.S3_BUCKET or settings.AWS_S3_BUCKET or "").strip()
    if not bucket:
        raise S3StorageError(
            "S3 bucket is not configured. Set AWS_S3_BUCKET (or S3_BUCKET) "
            "plus AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION."
        )
    return bucket


def _region() -> str:
    return (
        (settings.S3_REGION or "").strip()
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def base_prefix() -> str:
    """Root folder inside the bucket (default ``NoteBase/``)."""
    prefix = (settings.S3_BASE_PREFIX or "NoteBase").strip().strip("/") or "NoteBase"
    return f"{prefix}/"


def validate_configuration(*, probe: bool = False) -> dict[str, Any]:
    """Return (and log once) non-secret S3 config; optionally probe the bucket.

    Never logs credentials. Raises ``S3StorageError`` when configured but invalid.
    """
    global _config_logged

    info: dict[str, Any] = {
        "configured": is_configured(),
        "bucket": (settings.S3_BUCKET or settings.AWS_S3_BUCKET or "").strip() or None,
        "base_prefix": base_prefix() or None,
        "region": _region(),
        "endpoint_url": (settings.S3_ENDPOINT_URL or "").strip() or None,
        "has_access_key_env": bool(os.getenv("AWS_ACCESS_KEY_ID")),
        "has_secret_key_env": bool(os.getenv("AWS_SECRET_ACCESS_KEY")),
    }

    if not info["configured"]:
        if not _config_logged:
            logger.info("S3 storage disabled — AWS_S3_BUCKET / S3_BUCKET not set")
            _config_logged = True
        return info

    if not _config_logged:
        logger.info(
            "S3 storage enabled: bucket=%s prefix=%s region=%s endpoint=%s",
            info["bucket"],
            info["base_prefix"],
            info["region"],
            info["endpoint_url"] or "aws",
        )
        _config_logged = True

    if probe:
        try:
            _client().head_bucket(Bucket=_bucket())
            info["reachable"] = True
        except S3StorageError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise S3StorageError(
                f"S3 bucket {_bucket()!r} is not reachable in region {_region()}: {exc}"
            ) from exc

    return info


def _client():
    """Return a cached boto3 S3 client built from the ambient AWS credentials."""
    global _cached_client, _cached_bucket

    bucket = _bucket()

    with _client_lock:
        if _cached_client is not None and _cached_bucket == bucket:
            return _cached_client

        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover — boto3 ships in requirements
            raise S3StorageError(
                "boto3 is required for S3 storage. Install with: pip install boto3"
            ) from exc

        s3_config: dict[str, Any] = {}
        if settings.S3_SERVER_SIDE_ENCRYPTION:
            s3_config["ServerSideEncryption"] = settings.S3_SERVER_SIDE_ENCRYPTION

        config_kwargs: dict[str, Any] = {
            "region_name": _region(),
            "config": Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        }
        if s3_config:
            config_kwargs["config"] = Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
                s3=s3_config,
            )
        if settings.S3_ENDPOINT_URL:
            # MinIO / LocalStack compatible endpoints.
            config_kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL

        # Explicit env credentials are preferred when present; otherwise boto3
        # falls through to the shared credentials file / IAM role chain.
        access_key = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
        secret_key = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
        session_token = (os.getenv("AWS_SESSION_TOKEN") or "").strip() or None
        if access_key and secret_key:
            config_kwargs["aws_access_key_id"] = access_key
            config_kwargs["aws_secret_access_key"] = secret_key
            if session_token:
                config_kwargs["aws_session_token"] = session_token
        elif access_key or secret_key:
            raise S3StorageError(
                "Both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set together."
            )

        _cached_client = boto3.client("s3", **config_kwargs)
        _cached_bucket = bucket
        return _cached_client


def reset_client_cache() -> None:
    """Drop the cached client (used by tests and after credential rotation)."""
    global _cached_client, _cached_bucket, _config_logged
    with _client_lock:
        _cached_client = None
        _cached_bucket = None
        _config_logged = False


def _is_missing(exc: Exception) -> bool:
    response = getattr(exc, "response", None) or {}
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"NoSuchKey", "NotFound", "404", "NoSuchBucket"} or status == 404


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return None


# ── Key layout ───────────────────────────────────────────────────────────────

def _join_under_base(*parts: str) -> str:
    """Join path segments under the optional base prefix, rejecting unsafe segments."""
    cleaned: list[str] = []
    for part in parts:
        for segment in str(part).replace("\\", "/").split("/"):
            segment = segment.strip()
            if not segment or segment in {".", ".."}:
                continue
            cleaned.append(segment)
    return f"{base_prefix()}{'/'.join(cleaned)}"


def pdf_key(user_id: str, filename: str) -> str:
    """Object key: ``NoteBase/{user_id}/pdfs/{filename}``."""
    safe_name = Path(str(filename)).name  # blocks ../ traversal from stored names
    return _join_under_base(str(user_id), settings.S3_PDFS_PREFIX, safe_name)


def notes_key(user_id: str) -> str:
    """Object key: ``NoteBase/{user_id}/sticky_notes/notes.json``."""
    return _join_under_base(str(user_id), settings.S3_NOTES_PREFIX, NOTES_FILENAME)


def folder_key(relative_path: str) -> str:
    """Folder marker key under the user/base prefix, always ending with ``/``."""
    key = _join_under_base(relative_path)
    return key if key.endswith("/") else f"{key}/"


def _put_args(extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    args: dict[str, Any] = dict(extra or {})
    if settings.S3_SERVER_SIDE_ENCRYPTION:
        args["ServerSideEncryption"] = settings.S3_SERVER_SIDE_ENCRYPTION
    return args


def create_folder(relative_path: str) -> str:
    """Create an empty folder marker under the bucket root / optional base prefix.

    ``relative_path`` is relative to the base prefix (e.g. ``{user_id}/pdfs``).
    Returns the full object key ending in ``/``.
    """
    path = str(relative_path or "").strip().strip("/")
    if not path:
        raise S3StorageError("Folder path is required")

    for segment in path.replace("\\", "/").split("/"):
        if not segment or segment in {".", ".."} or not _FOLDER_SEGMENT_RE.match(segment):
            raise S3StorageError(
                f"Invalid folder segment {segment!r}. "
                "Use letters, numbers, dots, underscores, or hyphens."
            )

    key = folder_key(path)
    try:
        _client().put_object(
            Bucket=_bucket(),
            Key=key,
            Body=b"",
            ContentType="application/x-directory",
            **_put_args(),
        )
    except S3StorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise S3StorageError(f"Could not create folder {key}: {exc}") from exc

    logger.info("Created S3 folder %s", key)
    return key


def list_prefix(
    relative_path: str = "",
    *,
    max_keys: int = 200,
) -> list[dict[str, Any]]:
    """List objects and common prefixes under ``{base}{relative_path}``."""
    prefix = folder_key(relative_path) if relative_path.strip() else base_prefix()
    try:
        response = _client().list_objects_v2(
            Bucket=_bucket(),
            Prefix=prefix,
            Delimiter="/",
            MaxKeys=max(1, min(int(max_keys), 1000)),
        )
    except S3StorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise S3StorageError(f"Could not list {prefix}: {exc}") from exc

    items: list[dict[str, Any]] = []
    for entry in response.get("CommonPrefixes") or []:
        key = entry.get("Prefix") or ""
        items.append({"key": key, "type": "folder", "size": 0, "last_modified": None})
    for entry in response.get("Contents") or []:
        key = entry.get("Key") or ""
        if key.endswith("/"):
            items.append(
                {
                    "key": key,
                    "type": "folder",
                    "size": 0,
                    "last_modified": _iso(entry.get("LastModified")),
                }
            )
            continue
        items.append(
            {
                "key": key,
                "type": "file",
                "size": int(entry.get("Size") or 0),
                "last_modified": _iso(entry.get("LastModified")),
            }
        )
    return items


def ensure_user_folders(user_id: str) -> list[str]:
    """Ensure the user's ``pdfs/`` and ``sticky_notes/`` folders exist."""
    uid = str(user_id).strip()
    if not uid:
        raise S3StorageError("user_id is required to create storage folders")
    return [
        create_folder(f"{uid}/{settings.S3_PDFS_PREFIX}"),
        create_folder(f"{uid}/{settings.S3_NOTES_PREFIX}"),
    ]


# ── PDF objects ──────────────────────────────────────────────────────────────

def upload_pdf(user_id: str, filename: str, local_path: str) -> str:
    """Upload a local file to the user's ``pdfs/`` folder. Returns the object key."""
    key = pdf_key(user_id, filename)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    try:
        _client().upload_file(
            str(local_path),
            _bucket(),
            key,
            ExtraArgs=_put_args({"ContentType": content_type}),
        )
    except S3StorageError:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced as a typed storage error
        raise S3StorageError(f"Could not upload {key}: {exc}") from exc
    logger.info("Stored document in S3 for user %s at %s", user_id, key)
    return key


def open_pdf(user_id: str, filename: str, *, key: str | None = None):
    """Return a context manager over the streamed object body for a stored document.

    Usage::

        with s3_storage.open_pdf(user_id, filename) as body:
            data = body.read()

    Returns ``None`` when the object is missing. Raises ``S3StorageError`` for
    non-transient failures. Pass ``key`` when the database already stores the
    exact object key.
    """
    object_key = (key or "").strip() or pdf_key(user_id, filename)
    try:
        response = _client().get_object(Bucket=_bucket(), Key=object_key)
    except S3StorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_missing(exc):
            return None
        raise S3StorageError(f"Could not read {object_key}: {exc}") from exc

    body = response["Body"]
    content_length = int(response.get("ContentLength") or 0)

    class _StreamProxy:
        def __init__(self, body, content_length):
            self._body = body
            self.content_length = content_length

        def __enter__(self):
            return self

        def __exit__(self, *args):
            try:
                close = getattr(self._body, "close", None)
                if close is not None:
                    close()
            except Exception:
                pass

        def read(self, *args, **kwargs):
            return self._body.read(*args, **kwargs)

        def iter_chunks(self, chunk_size=8192):
            return self._body.iter_chunks(chunk_size=chunk_size)

    return _StreamProxy(body, content_length)


def download_pdf_to_path(
    user_id: str,
    filename: str,
    destination: str,
    *,
    key: str | None = None,
) -> bool:
    """Download a stored document to ``destination``; False when it is missing."""
    object_key = (key or "").strip() or pdf_key(user_id, filename)
    try:
        _client().download_file(_bucket(), object_key, str(destination))
        return True
    except S3StorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_missing(exc):
            return False
        raise S3StorageError(f"Could not download {object_key}: {exc}") from exc


def delete_pdf(user_id: str, filename: str, *, key: str | None = None) -> bool:
    """Delete a stored document. Missing objects are treated as success."""
    if not is_configured():
        return False
    object_key = (key or "").strip() or pdf_key(user_id, filename)
    try:
        _client().delete_object(Bucket=_bucket(), Key=object_key)
        return True
    except S3StorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise S3StorageError(f"Could not delete {object_key}: {exc}") from exc


# ── Sticky notes ─────────────────────────────────────────────────────────────

def load_notes(user_id: str) -> Tuple[Optional[dict], Optional[str]]:
    """Load the user's CKEditor JSON document.

    Returns ``(document, updated_at)``. ``(None, None)`` means the user has not
    saved notes yet. A corrupt/legacy object is reported as "no notes" and
    logged rather than breaking the editor.
    """
    key = notes_key(user_id)
    try:
        response = _client().get_object(Bucket=_bucket(), Key=key)
        raw = response["Body"].read()
    except S3StorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_missing(exc):
            return None, None
        raise S3StorageError(f"Could not load {key}: {exc}") from exc

    updated_at = _iso(response.get("LastModified")) or (
        response.get("Metadata") or {}
    ).get("updated-at")

    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Sticky notes for user %s are not valid JSON: %s", user_id, exc)
        return None, updated_at

    if not isinstance(document, dict):
        logger.warning(
            "Sticky notes for user %s have an unexpected JSON root (%s)",
            user_id,
            type(document).__name__,
        )
        return None, updated_at

    return document, updated_at


def save_notes(user_id: str, document: dict) -> str:
    """Persist a CKEditor JSON document, creating the file if needed."""
    if not isinstance(document, dict):
        raise S3StorageError("Sticky notes must be a JSON object")

    key = notes_key(user_id)
    payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    saved_at = datetime.now(timezone.utc).isoformat()
    try:
        _client().put_object(
            Bucket=_bucket(),
            Key=key,
            Body=payload,
            ContentType=_NOTES_CONTENT_TYPE,
            CacheControl="no-store",
            Metadata={"updated-at": saved_at, "user-id": str(user_id)},
            **{k: v for k, v in _put_args().items()},
        )
    except S3StorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise S3StorageError(f"Could not save {key}: {exc}") from exc

    logger.info("Saved sticky notes for user %s (%d bytes)", user_id, len(payload))
    return saved_at


def delete_notes(user_id: str) -> bool:
    """Remove the user's notes object. Missing objects are treated as success."""
    if not is_configured():
        return False
    key = notes_key(user_id)
    try:
        _client().delete_object(Bucket=_bucket(), Key=key)
    except S3StorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise S3StorageError(f"Could not delete {key}: {exc}") from exc
    return True
