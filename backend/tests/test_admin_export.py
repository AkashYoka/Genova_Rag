"""
Unit tests for the secure admin database export endpoint (#437).
"""
import pytest
from fastapi.testclient import TestClient
from app.models import User
from app.auth import create_access_token


@pytest.fixture()
def admin_auth_headers(db_session):
    """Create a temporary authenticated administrator session context."""
    admin_user = User(
        username="root_admin",
        email="admin@enterprise.rag",
        hashed_password="securepassword",
        role="admin",
    )
    db_session.add(admin_user)
    db_session.commit()
    db_session.refresh(admin_user)
    token = create_access_token(admin_user.id)
    return {"Authorization": f"Bearer {token}"}


def test_export_db_enforces_strict_admin_restriction(client: TestClient, auth_headers):
    """Ensure standard authenticated non-admin users are strictly rejected with a 403."""
    response = client.get("/api/v1/admin/export-db?format=json", headers=auth_headers)
    assert response.status_code == 403


def test_export_db_json_format_success(client: TestClient, admin_auth_headers):
    """Verify administrator can pull back entire schema state as an organized JSON object."""
    response = client.get("/api/v1/admin/export-db?format=json", headers=admin_auth_headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "attachment; filename=db_backup_" in response.headers["content-disposition"]
    assert response.headers["x-content-type-options"] == "nosniff"

    data = response.json()
    assert isinstance(data, dict)
    assert "users" in data


def test_export_db_sql_format_success(client: TestClient, admin_auth_headers):
    """Verify administrator can pull back sequential structural SQL statements."""
    response = client.get("/api/v1/admin/export-db?format=sql", headers=admin_auth_headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/sql")
    assert "attachment; filename=db_backup_" in response.headers["content-disposition"]

    sql_text = response.text
    assert "Database Backup" in sql_text
    assert "INSERT INTO" in sql_text


def test_export_db_invalid_format_parameter_rejection(client: TestClient, admin_auth_headers):
    """Verify endpoint terminates cycle elegantly with a 400 when an unmapped format is requested."""
    response = client.get("/api/v1/admin/export-db?format=yaml", headers=admin_auth_headers)
    assert response.status_code == 400
    assert "Invalid export format" in response.json()["detail"]


def test_export_db_redacts_credential_columns(client: TestClient, admin_auth_headers, db_session):
    """Credential-bearing columns must never appear in cleartext admin backups."""
    from app.models import User

    admin = db_session.query(User).filter(User.email == "admin@enterprise.rag").one()
    admin.google_refresh_token = "ya29.secret-refresh-token"
    admin.hashed_password = "$2b$12$should-never-leak"
    db_session.commit()

    response = client.get("/api/v1/admin/export-db?format=json", headers=admin_auth_headers)
    assert response.status_code == 200
    payload = response.json()
    users = payload.get("users") or []
    assert users, "expected users table in export"
    row = next(u for u in users if u.get("email") == "admin@enterprise.rag")
    assert row.get("hashed_password") == "[REDACTED]"
    assert row.get("google_refresh_token") == "[REDACTED]"
    assert "ya29.secret-refresh-token" not in response.text
    assert "$2b$12$should-never-leak" not in response.text


def test_should_redact_export_column_heuristics():
    from app.routes.admin import _should_redact_export_column

    assert _should_redact_export_column("hashed_password")
    assert _should_redact_export_column("google_refresh_token")
    assert _should_redact_export_column("credentials_json")
    assert _should_redact_export_column("oauth_access_token")
    assert not _should_redact_export_column("token_version")
    assert not _should_redact_export_column("username")
