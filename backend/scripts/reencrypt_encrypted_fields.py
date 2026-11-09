#!/usr/bin/env python3
"""Re-encrypt EncryptedString columns after FIELD_ENCRYPTION_KEY rotation.

Prerequisites:
  FIELD_ENCRYPTION_KEY=<new>
  FIELD_ENCRYPTION_KEY_PREVIOUS=<old>

Run against a maintenance window after deploy with both keys configured.
Does NOT bump token_version / rotate SECRET_KEY — do that in a separate step
per SECRET_ROTATION_RUNBOOK.md after encryption migration is verified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal  # noqa: E402
from app.models import User  # noqa: E402
from app.security.encryption import clear_fernet_cache, decrypt_field, encrypt_field  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    clear_fernet_cache()
    db = SessionLocal()
    updated = 0
    skipped = 0
    try:
        users = db.query(User).filter(User.google_refresh_token.isnot(None)).all()
        for user in users:
            raw = user.google_refresh_token
            if not raw:
                skipped += 1
                continue
            # ORM already decrypted via key ring. Re-assign to force re-encrypt
            # with the primary key on flush.
            plain = raw
            if args.dry_run:
                # Verify decrypt path works
                if decrypt_field(encrypt_field(plain)) != plain:
                    print(f"FAIL verify user={user.id}")
                    return 1
                updated += 1
                continue
            user.google_refresh_token = plain
            updated += 1
        if not args.dry_run:
            db.commit()
        print(f"{'dry-run ' if args.dry_run else ''}reencrypted={updated} skipped={skipped}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
