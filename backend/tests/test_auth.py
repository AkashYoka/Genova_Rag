from datetime import datetime, timedelta, timezone

import jwt

from app.config import get_settings


VALID_TEST_PASSWORD = "Password1!"


def test_register_success(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "newuser",
            "email": "newuser@example.com",
            "password": VALID_TEST_PASSWORD,
        },
    )

    assert response.status_code == 201
    payload = response.json()
    # REQUIRE_EMAIL_VERIFICATION defaults to false (POC): login immediately.
    assert payload["message"] == "Registration successful. You can log in now."
    assert payload["email"] == "newuser@example.com"
    assert payload.get("verification_url") in (None, "")

    login = client.post(
        "/api/v1/auth/login",
        json={"email": "newuser@example.com", "password": VALID_TEST_PASSWORD},
    )
    assert login.status_code == 200
    assert "access_token" in login.json() or login.json().get("user", {}).get("email") == "newuser@example.com"


def test_register_duplicate_email_or_username_conflict(client):
    payload = {
        "username": "dupuser",
        "email": "dup@example.com",
        "password": VALID_TEST_PASSWORD,
    }
    first = client.post("/api/v1/auth/register", json=payload)
    assert first.status_code == 201

    duplicate_email = client.post(
        "/api/v1/auth/register",
        json={**payload, "username": "anotheruser"},
    )
    assert duplicate_email.status_code == 409
    assert duplicate_email.json()["error"]["message"] == "Email already registered"

    duplicate_username = client.post(
        "/api/v1/auth/register",
        json={**payload, "email": "another@example.com"},
    )
    assert duplicate_username.status_code == 409
    assert duplicate_username.json()["error"]["message"] == "Username already taken"


def test_register_rejects_weak_password(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "weakpassuser",
            "email": "weakpass@example.com",
            "password": "123456",
        },
    )

    assert response.status_code == 422
    errors = response.json()["error"]["details"]["errors"]
    messages = " ".join(item["message"] for item in errors)
    assert "uppercase" in messages.lower() or "8 characters" in messages.lower()


def test_register_rejects_password_missing_special_character(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "specialcharuser",
            "email": "specialchar@example.com",
            "password": "Password1",
        },
    )

    assert response.status_code == 422
    errors = response.json()["error"]["details"]["errors"]
    messages = " ".join(item["message"] for item in errors).lower()
    assert "special character" in messages


def test_login_success(client, user):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "password123"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["access_token"]
    assert payload["refresh_token"]
    assert payload["user"]["username"] == user.username


def test_login_invalid_password(client, user):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "wrong-password"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid email or password"


def test_login_invalid_email(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "missing@example.com", "password": "password123"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid email or password"


def test_auth_me_success(client, auth_headers, user):
    response = client.get("/api/v1/auth/me", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == str(user.id)
    assert payload["username"] == user.username
    assert payload["email"] == user.email


def test_auth_me_requires_auth(client):
    response = client.get("/api/v1/auth/me")

    assert response.status_code in (401, 403)


def test_auth_me_rejects_expired_token(client, user):
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expired_token = jwt.encode(
        {
            "sub": str(user.id),
            "type": "access",
            "exp": now - timedelta(minutes=1),
            "iat": now - timedelta(minutes=2),
        },
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired token"


def test_refresh_token_success(client, refresh_token):
    response = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["access_token"]
    assert payload["refresh_token"]
    assert payload["token_type"] == "bearer"


def test_logout_invalidates_tokens(client, user):
    login = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "password123"},
    )
    assert login.status_code == 200
    tokens = login.json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    me_ok = client.get("/api/v1/auth/me", headers=headers)
    assert me_ok.status_code == 200

    logout = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": tokens["refresh_token"]},
        headers=headers,
    )
    assert logout.status_code == 200

    me_after = client.get("/api/v1/auth/me", headers=headers)
    assert me_after.status_code == 401

    refresh = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert refresh.status_code == 401

def test_update_user_info_rejects_duplicate_email(client, auth_headers, other_user):
    response = client.put(
        "/api/v1/auth/update",
        json={"email": other_user.email},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Email already exists"


def test_google_drive_connect_returns_auth_url(client, auth_headers, monkeypatch):
    from app.routes import auth as auth_routes

    class FakeFlow:
        def authorization_url(self, **kwargs):
            assert kwargs["access_type"] == "offline"
            assert kwargs["prompt"] == "consent"
            return "https://accounts.google.com/o/oauth2/auth?state=signed-state", "signed-state"

    monkeypatch.setattr(auth_routes.settings, "GOOGLE_CLIENT_ID", "google-client-id")
    monkeypatch.setattr(auth_routes.settings, "GOOGLE_CLIENT_SECRET", "google-client-secret")
    monkeypatch.setattr(auth_routes, "_google_drive_flow", lambda state: FakeFlow())

    response = client.get("/api/v1/auth/google-drive/connect", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["auth_url"].startswith("https://accounts.google.com")


def test_google_drive_status_reflects_stored_token(client, auth_headers, user, db_session):
    initial = client.get("/api/v1/auth/google-drive/status", headers=auth_headers)
    assert initial.status_code == 200
    assert initial.json() == {"connected": False}

    user.google_refresh_token = "google-refresh-token"
    db_session.commit()

    connected = client.get("/api/v1/auth/google-drive/status", headers=auth_headers)
    assert connected.status_code == 200
    assert connected.json() == {"connected": True}


def test_google_drive_callback_stores_encrypted_refresh_token(client, user, db_session, monkeypatch):
    from app.routes import auth as auth_routes
    from sqlalchemy import text

    class FakeCredentials:
        refresh_token = "google-refresh-token"

    class FakeFlow:
        credentials = FakeCredentials()

        def fetch_token(self, code):
            assert code == "oauth-code"

    monkeypatch.setattr(auth_routes.settings, "GOOGLE_CLIENT_ID", "google-client-id")
    monkeypatch.setattr(auth_routes.settings, "GOOGLE_CLIENT_SECRET", "google-client-secret")
    monkeypatch.setattr(auth_routes, "_google_drive_flow", lambda state: FakeFlow())

    state = auth_routes._create_google_drive_state(user.id)
    response = client.get(
        "/api/v1/auth/google-drive/callback",
        params={"code": "oauth-code", "state": state},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert "Google Drive connected" in response.text

    db_session.refresh(user)
    assert user.google_refresh_token == "google-refresh-token"

    row = db_session.execute(text("SELECT google_refresh_token FROM users WHERE id = :id"), {"id": user.id}).fetchone()
    stored_token = row[0]
    assert stored_token is not None
    assert stored_token != "google-refresh-token"


def test_google_drive_disconnect_removes_token(client, auth_headers, user, db_session):
    user.google_refresh_token = "google-refresh-token"
    db_session.commit()

    response = client.delete("/api/v1/auth/google-drive/disconnect", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {"connected": False}

    db_session.refresh(user)
    assert user.google_refresh_token is None
