"""Fernet field encryption with primary + previous key support.

During FIELD_ENCRYPTION_KEY rotation:
1. Set FIELD_ENCRYPTION_KEY to the new key.
2. Set FIELD_ENCRYPTION_KEY_PREVIOUS to the old key.
3. Deploy (reads try primary then previous; writes use primary).
4. Run the re-encrypt utility / admin job.
5. Remove FIELD_ENCRYPTION_KEY_PREVIOUS after verification.
6. Only then bump token_version / invalidate sessions if SECRET_KEY also rotated.
"""
from __future__ import annotations

import base64
import hashlib
import logging
from functools import lru_cache
from typing import List, Optional

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from app.config import get_settings

logger = logging.getLogger(__name__)


def _fernet_from_material(material: str) -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(material.encode("utf-8")).digest())
    return Fernet(key)


@lru_cache(maxsize=4)
def get_fernet_keys(primary: str, previous: str = "") -> MultiFernet:
    """Build a MultiFernet ring: encrypt with primary, decrypt with primary then previous."""
    fernets: List[Fernet] = [_fernet_from_material(primary)]
    prev = (previous or "").strip()
    if prev and prev != primary:
        fernets.append(_fernet_from_material(prev))
    return MultiFernet(fernets)


def _ring() -> MultiFernet:
    settings = get_settings()
    primary = settings.FIELD_ENCRYPTION_KEY
    previous = getattr(settings, "FIELD_ENCRYPTION_KEY_PREVIOUS", "") or ""
    return get_fernet_keys(primary, previous)


def encrypt_field(value: str) -> str:
    return _ring().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_field(value: str) -> Optional[str]:
    try:
        return _ring().decrypt(value.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        logger.warning("Field decryption failed with configured key ring")
        return None
    except Exception:
        logger.exception("Unexpected field decryption error")
        return None


def reencrypt_ciphertext(value: str) -> Optional[str]:
    """Decrypt with the key ring and re-encrypt with the primary key."""
    plain = decrypt_field(value)
    if plain is None:
        return None
    return encrypt_field(plain)


def clear_fernet_cache() -> None:
    get_fernet_keys.cache_clear()
