from app.auth import create_access_token, create_invite_token, hash_password
from app.models import User, Workspace, WorkspaceInvitation, WorkspaceMember, WorkspaceRole
import hashlib
from datetime import datetime, timedelta, timezone


def test_workspace_invite_requires_auth(client, db_session, user):
    token = create_access_token(user.id)
    response = client.post(
        "/api/v1/workspaces/invite",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": "invitee@example.com", "workspace_id": "missing"},
    )
    # Non-admin without membership → 403/404
    assert response.status_code in {403, 404}


def test_workspace_invite_creates_invitation_and_sends_email(client, db_session, monkeypatch):
    admin = User(
        username="admin",
        email="admin@example.com",
        hashed_password=hash_password("password123"),
        is_admin=True,
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)

    workspace = Workspace(name="Engineering", created_by=admin.id)
    db_session.add(workspace)
    db_session.commit()
    db_session.refresh(workspace)

    sent = {}

    def fake_send_workspace_invite_email(to, workspace_name, invite_link, expires_in_hours, personal_message=None):
        sent["to"] = to
        sent["workspace_name"] = workspace_name
        sent["invite_link"] = invite_link

    monkeypatch.setattr("app.routes.workspaces.send_workspace_invite_email", fake_send_workspace_invite_email)

    token = create_access_token(admin.id)
    response = client.post(
        "/api/v1/workspaces/invite",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": "invitee@example.com", "workspace_id": str(workspace.id)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["email"] == "invitee@example.com"
    assert payload["workspace_id"] == str(workspace.id)
    assert payload["workspace_name"] == "Engineering"
    assert payload.get("invite_link") == ""
    assert sent["to"] == "invitee@example.com"
    assert sent["workspace_name"] == "Engineering"
    assert "token=" in sent["invite_link"]

    invitation = db_session.query(WorkspaceInvitation).filter_by(email="invitee@example.com").first()
    assert invitation is not None
    assert str(invitation.workspace_id) == str(workspace.id)


def test_workspace_invite_does_not_enumerate_existing_emails(client, db_session, monkeypatch, user):
    admin = User(
        username="admin2",
        email="admin2@example.com",
        hashed_password=hash_password("password123"),
        is_admin=True,
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)

    workspace = Workspace(name="Engineering", created_by=admin.id)
    db_session.add(workspace)
    db_session.commit()
    db_session.refresh(workspace)

    sent = {}

    def fake_send(*args, **kwargs):
        sent["called"] = True

    monkeypatch.setattr("app.routes.workspaces.send_workspace_invite_email", fake_send)

    token = create_access_token(admin.id)
    response = client.post(
        "/api/v1/workspaces/invite",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": user.email, "workspace_id": str(workspace.id)},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["email"] == user.email
    assert payload.get("invite_link") == ""
    assert not sent.get("called")
    assert (
        db_session.query(WorkspaceInvitation).filter_by(email=user.email).first()
        is None
    )


def test_workspace_invite_accept_binds_workspace_id(client, db_session, user):
    admin = User(
        username="admin3",
        email="admin3@example.com",
        hashed_password=hash_password("password123"),
        is_admin=True,
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)

    ws_a = Workspace(name="Engineering", created_by=admin.id)
    ws_b = Workspace(name="Engineering", created_by=admin.id)
    db_session.add_all([ws_a, ws_b])
    db_session.commit()
    db_session.refresh(ws_a)
    db_session.refresh(ws_b)

    raw = create_invite_token(admin.id, user.email, workspace_id=str(ws_a.id), workspace_name="Engineering")
    invitation = WorkspaceInvitation(
        email=user.email,
        inviter_id=admin.id,
        token_hash=hashlib.sha256(raw.encode()).hexdigest(),
        workspace_id=ws_a.id,
        workspace_name="Engineering",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )
    db_session.add(invitation)
    db_session.commit()

    token = create_access_token(user.id)
    response = client.post(
        "/api/v1/workspaces/invitations/accept",
        headers={"Authorization": f"Bearer {token}"},
        json={"token": raw},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["workspace_id"] == str(ws_a.id)
    assert body["workspace_id"] != str(ws_b.id)

    member = (
        db_session.query(WorkspaceMember)
        .filter_by(workspace_id=ws_a.id, user_id=user.id)
        .first()
    )
    assert member is not None
    assert member.role == WorkspaceRole.viewer
