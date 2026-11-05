"""Promote existing localdev user to admin and set a known local password.

Never prints the password. Updates the existing row only (no duplicate users).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.auth import hash_password, verify_password  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import User, UserRole  # noqa: E402

EMAIL = "dev@genova.ai"
# Password supplied by operator for local development only.
PASSWORD = "localdev123"


def main() -> int:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == EMAIL).first()
        if not user:
            print(f"ERROR: user with email {EMAIL!r} not found")
            return 1

        user.role = UserRole.admin
        user.is_admin = True
        user.is_verified = True
        user.hashed_password = hash_password(PASSWORD)

        db.commit()
        db.refresh(user)

        ok = verify_password(PASSWORD, user.hashed_password)
        print(
            "UPDATED",
            f"id={user.id}",
            f"username={user.username}",
            f"email={user.email}",
            f"role={user.role}",
            f"is_admin={user.is_admin}",
            f"password_ok={ok}",
        )
        return 0 if ok else 2
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
