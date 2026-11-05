"""Admin dashboard authorization and operational endpoint coverage."""
from app.auth import create_access_token, hash_password, verify_password
from app.models import Document, User, UserRole


def _admin_headers(db_session):
    admin = User(
        username="dash_admin",
        email="dash_admin@example.com",
        hashed_password=hash_password("password123"),
        role=UserRole.admin,
        is_admin=True,
        is_verified=True,
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)
    return admin, {"Authorization": f"Bearer {create_access_token(admin.id)}"}


def test_promote_existing_user_to_admin_persists(db_session):
    """Updating an existing account must set role + is_admin without duplicates."""
    user = User(
        username="localdev",
        email="dev@genova.ai",
        hashed_password=hash_password("old-password"),
        role=UserRole.user,
        is_admin=False,
    )
    db_session.add(user)
    db_session.commit()
    user_id = user.id

    user.role = UserRole.admin
    user.is_admin = True
    user.is_verified = True
    user.hashed_password = hash_password("localdev123")
    db_session.commit()
    db_session.refresh(user)

    assert user.id == user_id
    assert user.role == UserRole.admin
    assert user.is_admin is True
    assert verify_password("localdev123", user.hashed_password)
    assert (
        db_session.query(User).filter(User.email == "dev@genova.ai").count() == 1
    )


def test_admin_endpoints_forbid_regular_user(client, auth_headers):
    paths = [
        "/api/v1/admin/stats",
        "/api/v1/admin/users",
        "/api/v1/admin/usage/summary",
        "/api/v1/admin/usage/daily",
        "/api/v1/admin/usage/events",
        "/api/v1/admin/documents",
        "/api/v1/admin/drive/connections",
        "/api/v1/admin/workspaces",
        "/api/v1/admin/api-keys/summary",
        "/api/v1/admin/export-db?format=json",
    ]
    for path in paths:
        response = client.get(path, headers=auth_headers)
        assert response.status_code == 403, path
        body = response.json()
        message = (
            body.get("error", {}).get("message")
            or body.get("detail")
            or ""
        )
        assert "Admin" in str(message)


def test_admin_documents_and_usage_ok(client, db_session, user):
    admin, headers = _admin_headers(db_session)
    db_session.add(
        Document(
            user_id=user.id,
            filename="fail.pdf",
            original_name="fail.pdf",
            file_size=10,
            status="failed",
            error_message="parse error",
        )
    )
    db_session.commit()

    docs = client.get("/api/v1/admin/documents?status=failed", headers=headers)
    assert docs.status_code == 200
    payload = docs.json()
    assert payload["total"] >= 1
    assert "failed" in payload["status_counts"]
    assert any(item["original_name"] == "fail.pdf" for item in payload["items"])

    usage = client.get("/api/v1/admin/usage/summary", headers=headers)
    assert usage.status_code == 200
    assert "request_count" in usage.json()
    assert "by_user" in usage.json()

    users = client.get("/api/v1/admin/users", headers=headers)
    assert users.status_code == 200
    emails = {row["email"] for row in users.json()}
    assert admin.email in emails
    assert user.email in emails

    drive = client.get("/api/v1/admin/drive/connections", headers=headers)
    assert drive.status_code == 200
    assert "items" in drive.json()

    workspaces = client.get("/api/v1/admin/workspaces", headers=headers)
    assert workspaces.status_code == 200

    keys = client.get("/api/v1/admin/api-keys/summary", headers=headers)
    assert keys.status_code == 200
    assert "total_keys" in keys.json()


def test_export_db_redacts_sensitive_columns(client, db_session):
    _, headers = _admin_headers(db_session)
    response = client.get("/api/v1/admin/export-db?format=json", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "users" in data
    for row in data["users"]:
        if "hashed_password" in row:
            assert row["hashed_password"] == "[REDACTED]"
