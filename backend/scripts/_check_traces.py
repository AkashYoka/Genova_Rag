"""Quick diagnostic: recent chats vs ai_traces."""
from __future__ import annotations

from sqlalchemy import text

from app.config import get_settings
from app.database import SessionLocal, engine
from sqlalchemy import inspect

s = get_settings()
print("DATABASE_URL", s.DATABASE_URL)
print("AI_TRACE_ENABLED", s.AI_TRACE_ENABLED)

insp = inspect(engine)
has = "ai_traces" in insp.get_table_names()
print("ai_traces_table", has)

db = SessionLocal()
try:
    if has:
        print("trace_count", db.execute(text("select count(*) from ai_traces")).scalar())
        for r in db.execute(
            text(
                "select trace_id, substr(question,1,60), status, created_at "
                "from ai_traces order by created_at desc limit 5"
            )
        ):
            print("trace", r)

    print("--- recent messages ---")
    for r in db.execute(
        text(
            "select id, role, substr(content,1,80), created_at, session_id "
            "from chat_messages order by created_at desc limit 10"
        )
    ):
        print(r)

    print("--- recent sessions ---")
    for r in db.execute(
        text(
            "select id, title, created_at from chat_sessions "
            "order by created_at desc limit 15"
        )
    ):
        print(r)
finally:
    db.close()
