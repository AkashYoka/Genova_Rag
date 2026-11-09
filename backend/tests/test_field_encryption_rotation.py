"""FIELD_ENCRYPTION_KEY primary + previous decrypt support."""
from __future__ import annotations

from app.security.encryption import (
    clear_fernet_cache,
    decrypt_field,
    encrypt_field,
    get_fernet_keys,
)


def test_previous_key_can_decrypt_during_rotation(monkeypatch):
    clear_fernet_cache()
    old = "old-field-encryption-key-material-32b!!"
    new = "new-field-encryption-key-material-32b!!"

    # Encrypt under old key
    monkeypatch.setenv("FIELD_ENCRYPTION_KEY", old)
    monkeypatch.setenv("FIELD_ENCRYPTION_KEY_PREVIOUS", "")
    from app.config import get_settings

    get_settings.cache_clear()
    clear_fernet_cache()
    # Use ring directly to avoid settings cache races
    old_ring = get_fernet_keys(old, "")
    ciphertext = old_ring.encrypt(b"secret-token").decode()

    # Decrypt with new primary + previous=old
    clear_fernet_cache()
    ring = get_fernet_keys(new, old)
    plain = ring.decrypt(ciphertext.encode()).decode()
    assert plain == "secret-token"

    # Re-encrypt under primary
    rotated = ring.encrypt(b"secret-token").decode()
    assert ring.decrypt(rotated.encode()).decode() == "secret-token"

    clear_fernet_cache()
    get_settings.cache_clear()
