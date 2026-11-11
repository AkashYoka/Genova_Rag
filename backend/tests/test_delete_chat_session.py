from app.models import ChatMessage, ChatSession, LLMUsage
from app.services.usage_tracker import record_llm_usage


def test_delete_chat_session_with_usage_rows(client, db_session, user, auth_headers):
    """Deleting a session must succeed even when llm_usage references it."""
    session = ChatSession(user_id=user.id, title="Delete me")
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    session_id = session.id

    user_msg = ChatMessage(
        user_id=user.id,
        session_id=session_id,
        role="user",
        content="hello",
    )
    assistant_msg = ChatMessage(
        user_id=user.id,
        session_id=session_id,
        role="assistant",
        content="hi",
    )
    db_session.add_all([user_msg, assistant_msg])
    db_session.commit()
    db_session.refresh(assistant_msg)

    row = record_llm_usage(
        db_session,
        user_id=str(user.id),
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "provider": "bedrock",
            "model_id": "us.amazon.nova-2-lite-v1:0",
        },
        usage_type="chat",
        session_id=str(session_id),
        message_id=str(assistant_msg.id),
        commit=True,
    )
    assert row is not None
    usage_id = row.id

    response = client.delete(
        f"/api/v1/chat/sessions/{session_id}",
        headers=auth_headers,
    )
    assert response.status_code == 204

    db_session.expire_all()
    assert (
        db_session.query(ChatSession).filter(ChatSession.id == session_id).first()
        is None
    )
    assert (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .count()
        == 0
    )

    kept = db_session.query(LLMUsage).filter(LLMUsage.id == usage_id).one()
    assert kept.session_id is None
    assert kept.message_id is None
    assert kept.total_tokens == 15
