from app.auth import create_access_token, hash_password
from app.models import ChatMessage, ChatSession, ResponseFeedback, User, UserRole


def _make_turn(db_session, user):
    session = ChatSession(user_id=user.id, title="Feedback test chat")
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)

    question = ChatMessage(
        user_id=user.id,
        session_id=session.id,
        role="user",
        content="What is the refund window for annual plans?",
    )
    response = ChatMessage(
        user_id=user.id,
        session_id=session.id,
        role="assistant",
        content="Annual plans can be refunded within 30 days of purchase.",
    )
    db_session.add_all([question, response])
    db_session.commit()
    db_session.refresh(question)
    db_session.refresh(response)
    return session, question, response


def _make_admin(db_session):
    admin = User(
        username="feedback-admin",
        email="feedback-admin@example.com",
        hashed_password=hash_password("password123"),
        is_admin=True,
        role=UserRole.admin,
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)
    return admin


def test_submit_feedback_persists_full_context(client, auth_headers, db_session, user):
    session, question, response = _make_turn(db_session, user)

    res = client.patch(
        f"/api/v1/chat/feedback/{response.id}",
        headers=auth_headers,
        json={
            "feedback": "down",
            "reason": "incomplete",
            "comment": "It does not mention the exceptions.",
        },
    )

    assert res.status_code == 200
    assert res.json()["feedback"] == "down"

    row = (
        db_session.query(ResponseFeedback)
        .filter(ResponseFeedback.message_id == response.id)
        .one()
    )
    assert row.user_id == user.id
    assert row.session_id == session.id
    assert row.question_message_id == question.id
    assert row.question == question.content
    assert row.response == response.content
    assert row.feedback == "down"
    assert row.reason == "incomplete"
    assert row.comment == "It does not mention the exceptions."


def test_feedback_cannot_be_submitted_twice(client, auth_headers, db_session, user):
    _session, _question, response = _make_turn(db_session, user)
    payload = {"feedback": "up", "reason": "helpful"}

    first = client.patch(
        f"/api/v1/chat/feedback/{response.id}",
        headers=auth_headers,
        json=payload,
    )
    second = client.patch(
        f"/api/v1/chat/feedback/{response.id}",
        headers=auth_headers,
        json=payload,
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert "already" in second.json()["error"]["message"].lower()
    assert (
        db_session.query(ResponseFeedback)
        .filter(ResponseFeedback.message_id == response.id)
        .count()
        == 1
    )


def test_feedback_rejects_unknown_reason(client, auth_headers, db_session, user):
    _session, _question, response = _make_turn(db_session, user)

    res = client.patch(
        f"/api/v1/chat/feedback/{response.id}",
        headers=auth_headers,
        json={"feedback": "down", "reason": "not-a-real-reason"},
    )

    assert res.status_code == 400
    assert "Unknown feedback reason" in res.json()["error"]["message"]


def test_feedback_is_owner_scoped(
    client, auth_headers, db_session, other_user_assistant_message
):
    res = client.patch(
        f"/api/v1/chat/feedback/{other_user_assistant_message.id}",
        headers=auth_headers,
        json={"feedback": "up"},
    )

    assert res.status_code == 404


def test_admin_feedback_requires_admin(client, auth_headers):
    res = client.get("/api/v1/admin/feedback", headers=auth_headers)

    assert res.status_code == 403


def test_admin_feedback_list_filters_and_summary(
    client, auth_headers, db_session, user
):
    _session, _question, response = _make_turn(db_session, user)
    client.patch(
        f"/api/v1/chat/feedback/{response.id}",
        headers=auth_headers,
        json={
            "feedback": "down",
            "reason": "not_relevant",
            "comment": "This is about a different product.",
        },
    )

    admin = _make_admin(db_session)
    admin_headers = {"Authorization": f"Bearer {create_access_token(admin.id)}"}

    res = client.get(
        "/api/v1/admin/feedback?feedback=down&search=refund",
        headers=admin_headers,
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["total"] == 1
    assert payload["summary"] == {
        "total": 1,
        "likes": 0,
        "dislikes": 1,
        "satisfaction_percentage": 0.0,
    }
    item = payload["items"][0]
    assert item["username"] == user.username
    assert item["feedback"] == "down"
    assert item["reason"] == "not_relevant"
    assert item["question"] == "What is the refund window for annual plans?"
    assert item["response"] == response.content
    assert item["session_id"] == _session.id
    assert item["message_id"] == response.id
    assert item["question_message_id"] == _question.id

    empty = client.get(
        "/api/v1/admin/feedback?feedback=up", headers=admin_headers
    )
    assert empty.status_code == 200
    assert empty.json()["total"] == 0
    assert empty.json()["summary"]["dislikes"] == 1


def test_admin_feedback_date_range(client, auth_headers, db_session, user):
    _session, _question, response = _make_turn(db_session, user)
    client.patch(
        f"/api/v1/chat/feedback/{response.id}",
        headers=auth_headers,
        json={"feedback": "up", "reason": "accurate"},
    )

    admin = _make_admin(db_session)
    admin_headers = {"Authorization": f"Bearer {create_access_token(admin.id)}"}

    in_range = client.get(
        "/api/v1/admin/feedback?start=2000-01-01&end=2999-12-31",
        headers=admin_headers,
    )
    out_of_range = client.get(
        "/api/v1/admin/feedback?end=2000-01-01", headers=admin_headers
    )

    assert in_range.status_code == 200
    assert in_range.json()["total"] == 1
    assert out_of_range.status_code == 200
    assert out_of_range.json()["total"] == 0
