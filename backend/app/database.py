"""
SQLAlchemy database setup — MySQL 8.4 LTS in deployments, SQLite by default
for local use. Uses synchronous SQLAlchemy for simplicity and compatibility.
"""
import os
import time
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Engine & Session ─────────────────────────────────
_database_url_lower = settings.DATABASE_URL.lower()
is_sqlite = _database_url_lower.startswith("sqlite")
is_mysql = _database_url_lower.startswith("mysql") or _database_url_lower.startswith("mariadb")

if is_sqlite:
    # ── Ensure data directory exists ─────────────────────
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    if db_path and os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},  # Required for SQLite
        echo=settings.DEBUG,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()
else:
    # MySQL 8.4 LTS (and any other server-backed dialect).  utf8mb4 keeps
    # document text, emoji and CJK intact; pool_recycle prevents "MySQL
    # server has gone away" on long-lived Celery workers.
    _connect_args = {"charset": "utf8mb4"} if is_mysql else {}
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_pre_ping=settings.DATABASE_POOL_PRE_PING,
        pool_recycle=settings.DATABASE_POOL_RECYCLE,  # Recycle stale connections
        connect_args=_connect_args,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()



def get_db():
    """FastAPI dependency — yields a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """Context manager for background tasks, streaming, and other uses outside FastAPI DI.

    Creates a new session, commits on success, rolls back SQLAlchemy errors
    (converting them to typed AppException), re-raises non-DB exceptions,
    and always closes the session.
    """
    from sqlalchemy.exc import SQLAlchemyError
    from app.exceptions import AppException

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise AppException(
            "DATABASE_ERROR",
            "A database error occurred while processing your request.",
            500,
            {"error": str(e)[:200]},
        ) from e
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

_SLOW_QUERY_THRESHOLD = settings.DATABASE_SLOW_QUERY_THRESHOLD


@event.listens_for(engine, "before_cursor_execute")
def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault("query_start_time", []).append(time.perf_counter())


@event.listens_for(engine, "after_cursor_execute")
def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    start_times = conn.info.get("query_start_time")
    if not start_times:
        return
    total_time = time.perf_counter() - start_times.pop(-1)
    if total_time > _SLOW_QUERY_THRESHOLD:
        logger.warning(
            "Slow query detected (%.2fs): %s",
            total_time,
            statement[:500],
        )


# ── Session Lifecycle Logging (DEBUG only) ───────────
if settings.DEBUG:
    @event.listens_for(SessionLocal, "after_begin")
    def _receive_after_begin(session, transaction, connection):
        logger.debug("Session %s began transaction", id(session))

    @event.listens_for(SessionLocal, "after_commit")
    def _receive_after_commit(session):
        logger.debug("Session %s committed", id(session))

    @event.listens_for(SessionLocal, "after_rollback")
    def _receive_after_rollback(session):
        logger.debug("Session %s rolled back", id(session))

    @event.listens_for(SessionLocal, "after_close")
    def _receive_after_close(session):
        logger.debug("Session %s closed", id(session))


def _ensure_index(table: str, index_name: str, columns: str) -> None:
    """Create a non-unique index when it is missing.

    MySQL has no ``CREATE INDEX IF NOT EXISTS``, so existence is checked via
    the inspector instead of relying on dialect-specific DDL.
    """
    try:
        inspector = inspect(engine)
        if table not in inspector.get_table_names():
            return
        existing = {index.get("name") for index in inspector.get_indexes(table)}
        if index_name in existing:
            return
        with engine.begin() as conn:
            conn.execute(text(f"CREATE INDEX {index_name} ON {table} ({columns})"))
        logger.info("Migration: created index %s", index_name)
    except Exception as exc:
        logger.warning("Index creation skipped for %s: %s", index_name, exc)


def _ensure_fk_on_delete_set_null(
    *,
    table: str,
    column: str,
    ref_table: str,
    constraint_name: str,
) -> None:
    """Ensure ``column`` references ``ref_table`` with ON DELETE SET NULL (MySQL).

    Existing databases may have been created without an ON DELETE clause, which
    can block or cascade document hard-deletes into chat history. Recreate the
    FK safely when needed.
    """
    if not is_mysql:
        return
    try:
        inspector = inspect(engine)
        if table not in inspector.get_table_names():
            return
        fks = inspector.get_foreign_keys(table)
        matching = [
            fk
            for fk in fks
            if column in (fk.get("constrained_columns") or [])
            and fk.get("referred_table") == ref_table
        ]
        already_ok = any(
            (fk.get("options") or {}).get("ondelete", "").upper() == "SET NULL"
            for fk in matching
        )
        if already_ok:
            return

        with engine.begin() as conn:
            for fk in matching:
                name = fk.get("name")
                if name:
                    conn.execute(
                        text(f"ALTER TABLE {table} DROP FOREIGN KEY `{name}`")
                    )
            conn.execute(
                text(
                    f"ALTER TABLE {table} "
                    f"ADD CONSTRAINT `{constraint_name}` "
                    f"FOREIGN KEY (`{column}`) REFERENCES `{ref_table}` (`id`) "
                    f"ON DELETE SET NULL"
                )
            )
        logger.info(
            "Migration: set ON DELETE SET NULL for %s.%s → %s.id",
            table,
            column,
            ref_table,
        )
    except Exception as exc:
        logger.warning(
            "FK ON DELETE SET NULL migration skipped for %s.%s: %s",
            table,
            column,
            exc,
        )


def _migrate_schema():
    """Apply schema migrations for existing databases (SQLite and MySQL).

    SQLAlchemy's ``create_all`` only creates new tables and does **not**
    add missing columns to existing tables.  This helper fills that gap
    for non-destructive changes such as new nullable columns.
    """
    inspector = inspect(engine)
    text_type = "LONGTEXT" if is_mysql else "TEXT"
    # DATETIME(6) keeps sub-second precision and avoids MySQL's 2038 TIMESTAMP limit.
    timestamp_type = "DATETIME(6)" if is_mysql else "TIMESTAMP"
    # Migrate users
    existing_users_columns = {c["name"] for c in inspector.get_columns("users")}
    users_migrations = [
        ("users", "google_refresh_token", f"ALTER TABLE users ADD COLUMN google_refresh_token {text_type}"),
        ("users", "role", "ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'user'"),
        ("users", "last_login", f"ALTER TABLE users ADD COLUMN last_login {timestamp_type}"),
        ("users", "is_verified", "ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT 1 NOT NULL"),
        ("users", "verification_token_hash", "ALTER TABLE users ADD COLUMN verification_token_hash VARCHAR(64)"),
        (
            "users",
            "verification_token_created_at",
            f"ALTER TABLE users ADD COLUMN verification_token_created_at {timestamp_type}",
        ),
        ("users", "display_name", "ALTER TABLE users ADD COLUMN display_name VARCHAR(120)"),
        ("users", "avatar_url", "ALTER TABLE users ADD COLUMN avatar_url VARCHAR(500)"),
        ("users", "token_version", "ALTER TABLE users ADD COLUMN token_version INTEGER DEFAULT 0 NOT NULL"),
        ("users", "dropbox_refresh_token", f"ALTER TABLE users ADD COLUMN dropbox_refresh_token {text_type}"),
        ("users", "dropbox_access_token", f"ALTER TABLE users ADD COLUMN dropbox_access_token {text_type}"),
        ("users", "dropbox_token_expires_at", f"ALTER TABLE users ADD COLUMN dropbox_token_expires_at {timestamp_type}"),
        ("users", "dropbox_account_id", "ALTER TABLE users ADD COLUMN dropbox_account_id VARCHAR(255)"),
        ("users", "dropbox_display_name", "ALTER TABLE users ADD COLUMN dropbox_display_name VARCHAR(255)"),
    ]
    for table, column, ddl in users_migrations:
        if column not in existing_users_columns:
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
                logger.info("Migration: added column %s.%s", table, column)
            except Exception as exc:
                logger.warning(
                    "Migration failed for %s.%s: %s", table, column, exc
                )

    # Migrate api_keys
    try:
        existing_keys_columns = {c["name"] for c in inspector.get_columns("api_keys")}
    except Exception:
        existing_keys_columns = set()
    keys_migrations = [
        ("api_keys", "name", "ALTER TABLE api_keys ADD COLUMN name VARCHAR(100) DEFAULT 'default'"),
        ("api_keys", "is_active", "ALTER TABLE api_keys ADD COLUMN is_active BOOLEAN DEFAULT 1 NOT NULL"),
        ("api_keys", "last_used_at", f"ALTER TABLE api_keys ADD COLUMN last_used_at {timestamp_type}"),
    ]
    for table, column, ddl in keys_migrations:
        if column not in existing_keys_columns:
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
                logger.info("Migration: added column %s.%s", table, column)
            except Exception as exc:
                logger.warning(
                    "Migration failed for %s.%s: %s", table, column, exc
                )

    # Migrate documents
    existing_docs_columns = {c["name"] for c in inspector.get_columns("documents")}
    docs_migrations = [
        ("documents", "last_accessed_at", f"ALTER TABLE documents ADD COLUMN last_accessed_at {timestamp_type}"),
        ("documents", "is_deleted", "ALTER TABLE documents ADD COLUMN is_deleted BOOLEAN DEFAULT 0 NOT NULL"),
        ("documents", "deleted_at", f"ALTER TABLE documents ADD COLUMN deleted_at {timestamp_type}"),
        ("documents", "summary", f"ALTER TABLE documents ADD COLUMN summary {text_type}"),
        ("documents", "chunk_size", "ALTER TABLE documents ADD COLUMN chunk_size INTEGER"),
        ("documents", "chunk_overlap", "ALTER TABLE documents ADD COLUMN chunk_overlap INTEGER"),
        ("documents", "drive_file_id", "ALTER TABLE documents ADD COLUMN drive_file_id VARCHAR(255)"),
        ("documents", "drive_folder_id", "ALTER TABLE documents ADD COLUMN drive_folder_id VARCHAR(255)"),
        ("documents", "drive_synced_at", f"ALTER TABLE documents ADD COLUMN drive_synced_at {timestamp_type}"),
        ("documents", "drive_modified_time", "ALTER TABLE documents ADD COLUMN drive_modified_time VARCHAR(64)"),
        ("documents", "source_type", "ALTER TABLE documents ADD COLUMN source_type VARCHAR(32) DEFAULT 'upload' NOT NULL"),
        ("documents", "external_file_id", "ALTER TABLE documents ADD COLUMN external_file_id VARCHAR(255)"),
        ("documents", "external_folder_id", "ALTER TABLE documents ADD COLUMN external_folder_id VARCHAR(255)"),
        ("documents", "external_revision", "ALTER TABLE documents ADD COLUMN external_revision VARCHAR(128)"),
        ("documents", "external_synced_at", f"ALTER TABLE documents ADD COLUMN external_synced_at {timestamp_type}"),
        ("documents", "lifecycle_status", "ALTER TABLE documents ADD COLUMN lifecycle_status VARCHAR(20) DEFAULT 'ACTIVE' NOT NULL"),
        ("documents", "source_deleted_at", f"ALTER TABLE documents ADD COLUMN source_deleted_at {timestamp_type}"),
        ("documents", "processing_progress", "ALTER TABLE documents ADD COLUMN processing_progress INTEGER DEFAULT 0"),
        ("documents", "processing_stage", "ALTER TABLE documents ADD COLUMN processing_stage VARCHAR(20) DEFAULT 'queued'"),
        ("documents", "retry_count", "ALTER TABLE documents ADD COLUMN retry_count INTEGER DEFAULT 0"),
        ("documents", "last_error_traceback", f"ALTER TABLE documents ADD COLUMN last_error_traceback {text_type}"),
        ("documents", "processing_started_at", f"ALTER TABLE documents ADD COLUMN processing_started_at {timestamp_type}"),
        ("documents", "completed_at", f"ALTER TABLE documents ADD COLUMN completed_at {timestamp_type}"),
        ("documents", "extracted_urls", f"ALTER TABLE documents ADD COLUMN extracted_urls {text_type}"),
        ("documents", "keywords", f"ALTER TABLE documents ADD COLUMN keywords {text_type}"),
        ("documents", "s3_key", "ALTER TABLE documents ADD COLUMN s3_key VARCHAR(512)"),
        ("documents", "storage_backend", "ALTER TABLE documents ADD COLUMN storage_backend VARCHAR(20) DEFAULT 'local'"),
        ("documents", "detected_domain", "ALTER TABLE documents ADD COLUMN detected_domain VARCHAR(64)"),
        ("documents", "domain_confidence", "ALTER TABLE documents ADD COLUMN domain_confidence FLOAT"),
    ]
    for table, column, ddl in docs_migrations:
        if column not in existing_docs_columns:
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
                logger.info("Migration: added column %s.%s", table, column)
            except Exception as exc:
                logger.warning(
                    "Migration failed for %s.%s: %s", table, column, exc
                )

    _ensure_index("documents", "ix_documents_source_type", "source_type")
    _ensure_index("documents", "ix_documents_external_file_id", "external_file_id")
    _ensure_index("documents", "ix_documents_external_folder_id", "external_folder_id")
    _ensure_index("documents", "ix_documents_lifecycle_status", "lifecycle_status")
    _ensure_index("users", "ix_users_dropbox_account_id", "dropbox_account_id")

    # Migrate chat_messages
    try:
        existing_chat_columns = {c["name"] for c in inspector.get_columns("chat_messages")}
    except Exception:
        existing_chat_columns = set()
    chat_migrations = [
        ("chat_messages", "feedback", "ALTER TABLE chat_messages ADD COLUMN feedback VARCHAR(10)"),
        ("chat_messages", "session_id", "ALTER TABLE chat_messages ADD COLUMN session_id CHAR(36)"),
    ]
    for table, column, ddl in chat_migrations:
        if column not in existing_chat_columns:
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
                logger.info("Migration: added column %s.%s", table, column)
            except Exception:
                logger.warning(
                    "Migration skipped (may already exist): %s.%s", table, column
                )
    
    # Ensure index exists on session_id
    _ensure_index("chat_messages", "ix_chat_messages_session_id", "session_id")
    _ensure_index(
        "chat_messages",
        "ix_chat_messages_session_user_created",
        "session_id, user_id, created_at",
    )

    # Document hard-delete must not erase chat history / usage rows.
    # Prefer ON DELETE SET NULL for document_id FKs (MySQL). SQLite tests rely on
    # create_all + explicit nulling in _hard_delete_document.
    if is_mysql:
        _ensure_fk_on_delete_set_null(
            table="chat_messages",
            column="document_id",
            ref_table="documents",
            constraint_name="fk_chat_messages_document_id",
        )
        _ensure_fk_on_delete_set_null(
            table="llm_usage",
            column="document_id",
            ref_table="documents",
            constraint_name="fk_llm_usage_document_id",
        )
        # Session/message deletes must keep usage ledger rows for billing history.
        _ensure_fk_on_delete_set_null(
            table="llm_usage",
            column="session_id",
            ref_table="chat_sessions",
            constraint_name="fk_llm_usage_session_id",
        )
        _ensure_fk_on_delete_set_null(
            table="llm_usage",
            column="message_id",
            ref_table="chat_messages",
            constraint_name="fk_llm_usage_message_id",
        )


    # Migrate documents — embedding cache tracking
    try:
        existing_docs_columns = {c["name"] for c in inspector.get_columns("documents")}
    except Exception:
        existing_docs_columns = set()

    embedding_cache_migrations = [
        (
            "documents",
            "extracted_urls",
            f"ALTER TABLE documents ADD COLUMN extracted_urls {text_type}",
        ),
    ]
    for table, column, ddl in embedding_cache_migrations:
        if column not in existing_docs_columns:
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
                logger.info("Migration: added column %s.%s", table, column)
            except Exception:
                logger.warning(
                    "Migration skipped (may already exist): %s.%s", table, column
                )


    # ── Workspace tables ──────────────────────────────────────────────────
    existing_tables = set(inspector.get_table_names())

    if "workspaces" not in existing_tables:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"""
                    CREATE TABLE workspaces (
                        id         CHAR(36)     PRIMARY KEY,
                        name       VARCHAR(255) NOT NULL,
                        created_by CHAR(36)     NOT NULL,
                        created_at {timestamp_type},
                        CONSTRAINT fk_workspaces_created_by
                            FOREIGN KEY (created_by) REFERENCES users (id)
                    )
                """))
            _ensure_index("workspaces", "ix_workspaces_created_by", "created_by")
            logger.info("Migration: created table workspaces")
        except Exception:
            logger.warning("Migration skipped (may already exist): workspaces")

    if "workspace_members" not in existing_tables:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"""
                    CREATE TABLE workspace_members (
                        id           CHAR(36)    PRIMARY KEY,
                        workspace_id CHAR(36)    NOT NULL,
                        user_id      CHAR(36)    NOT NULL,
                        role         VARCHAR(20) NOT NULL DEFAULT 'viewer',
                        joined_at    {timestamp_type},
                        CONSTRAINT uq_workspace_member UNIQUE (workspace_id, user_id),
                        CONSTRAINT fk_workspace_members_workspace
                            FOREIGN KEY (workspace_id) REFERENCES workspaces (id),
                        CONSTRAINT fk_workspace_members_user
                            FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """))
            _ensure_index(
                "workspace_members", "ix_workspace_members_workspace_id", "workspace_id"
            )
            _ensure_index("workspace_members", "ix_workspace_members_user_id", "user_id")
            logger.info("Migration: created table workspace_members")
        except Exception:
            logger.warning("Migration skipped (may already exist): workspace_members")

    if "workspace_invitations" not in existing_tables:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"""
                    CREATE TABLE workspace_invitations (
                        id             CHAR(36)     PRIMARY KEY,
                        email          VARCHAR(120) NOT NULL,
                        token_hash     VARCHAR(255) NOT NULL UNIQUE,
                        inviter_id     CHAR(36)     NOT NULL,
                        workspace_name VARCHAR(255) NOT NULL,
                        created_at     {timestamp_type},
                        expires_at     {timestamp_type}    NOT NULL,
                        accepted_at    {timestamp_type},
                        CONSTRAINT fk_workspace_invitations_inviter
                            FOREIGN KEY (inviter_id) REFERENCES users (id)
                    )
                """))
            _ensure_index(
                "workspace_invitations", "ix_workspace_invitations_email", "email"
            )
            _ensure_index(
                "workspace_invitations",
                "ix_workspace_invitations_token_hash",
                "token_hash",
            )
            _ensure_index(
                "workspace_invitations",
                "ix_workspace_invitations_inviter_id",
                "inviter_id",
            )
            logger.info("Migration: created table workspace_invitations")
        except Exception:
            logger.warning("Migration skipped (may already exist): workspace_invitations")

    # Bind invitations to workspace_id (FK) — never resolve by display name alone.
    try:
        inv_cols = {c["name"] for c in inspector.get_columns("workspace_invitations")}
    except Exception:
        inv_cols = set()
    if inv_cols and "workspace_id" not in inv_cols:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE workspace_invitations "
                        "ADD COLUMN workspace_id CHAR(36)"
                    )
                )
            _ensure_index(
                "workspace_invitations",
                "ix_workspace_invitations_workspace_id",
                "workspace_id",
            )
            logger.info("Migration: added column workspace_invitations.workspace_id")
        except Exception as exc:
            logger.warning(
                "Migration failed for workspace_invitations.workspace_id: %s", exc
            )


    # ── Drive connections table + columns ──────────────────────────────────
    existing_tables = set(inspector.get_table_names())
    if "drive_connections" not in existing_tables:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"""
                    CREATE TABLE drive_connections (
                        id                   CHAR(36)     PRIMARY KEY,
                        user_id              CHAR(36)     NOT NULL,
                        folder_id            VARCHAR(255) NOT NULL,
                        folder_name          VARCHAR(255),
                        folder_url           VARCHAR(500),
                        access_mode          VARCHAR(20)  NOT NULL DEFAULT 'public',
                        credentials_json     {text_type},
                        service_account_file VARCHAR(500),
                        enabled              BOOLEAN      NOT NULL DEFAULT 1,
                        sync_status          VARCHAR(20)  NOT NULL DEFAULT 'idle',
                        celery_task_id       VARCHAR(255),
                        last_error           {text_type},
                        last_synced_at       {timestamp_type},
                        created_at           {timestamp_type},
                        updated_at           {timestamp_type},
                        CONSTRAINT fk_drive_connections_user
                            FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """))
            _ensure_index(
                "drive_connections", "ix_drive_connections_user_id", "user_id"
            )
            _ensure_index(
                "drive_connections", "ix_drive_connections_folder_id", "folder_id"
            )
            logger.info("Migration: created table drive_connections")
        except Exception as exc:
            logger.warning("Migration skipped drive_connections: %s", exc)
    else:
        try:
            drive_cols = {c["name"] for c in inspector.get_columns("drive_connections")}
        except Exception:
            drive_cols = set()
        drive_migrations = [
            (
                "drive_connections",
                "folder_name",
                "ALTER TABLE drive_connections ADD COLUMN folder_name VARCHAR(255)",
            ),
            (
                "drive_connections",
                "folder_url",
                "ALTER TABLE drive_connections ADD COLUMN folder_url VARCHAR(500)",
            ),
            (
                "drive_connections",
                "sync_status",
                "ALTER TABLE drive_connections ADD COLUMN sync_status VARCHAR(20) DEFAULT 'idle' NOT NULL",
            ),
            (
                "drive_connections",
                "last_error",
                f"ALTER TABLE drive_connections ADD COLUMN last_error {text_type}",
            ),
            (
                "drive_connections",
                "access_mode",
                "ALTER TABLE drive_connections ADD COLUMN access_mode VARCHAR(20) DEFAULT 'public' NOT NULL",
            ),
            (
                "drive_connections",
                "celery_task_id",
                "ALTER TABLE drive_connections ADD COLUMN celery_task_id VARCHAR(255)",
            ),
        ]
        for table, column, ddl in drive_migrations:
            if column not in drive_cols:
                try:
                    with engine.begin() as conn:
                        conn.execute(text(ddl))
                    logger.info("Migration: added column %s.%s", table, column)
                except Exception as exc:
                    logger.warning(
                        "Migration failed for %s.%s: %s", table, column, exc
                    )

    # ── Dropbox connections + deletion tombstones ───────────────────────────
    existing_tables = set(inspector.get_table_names())
    if "dropbox_connections" not in existing_tables:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"""
                    CREATE TABLE dropbox_connections (
                        id                CHAR(36)      PRIMARY KEY,
                        user_id           CHAR(36)      NOT NULL,
                        connection_kind   VARCHAR(20)   NOT NULL DEFAULT 'oauth',
                        remote_folder_id  VARCHAR(255)  NOT NULL,
                        folder_path       VARCHAR(1024),
                        shared_link       VARCHAR(2048),
                        folder_name       VARCHAR(255),
                        folder_url        VARCHAR(2048),
                        pending_file_ids  {text_type},
                        enabled           BOOLEAN       NOT NULL DEFAULT 1,
                        sync_status       VARCHAR(20)   NOT NULL DEFAULT 'idle',
                        celery_task_id    VARCHAR(255),
                        last_error        {text_type},
                        last_synced_at    {timestamp_type},
                        created_at        {timestamp_type},
                        updated_at        {timestamp_type},
                        CONSTRAINT uq_dropbox_connection_user_folder
                            UNIQUE (user_id, connection_kind, remote_folder_id),
                        CONSTRAINT fk_dropbox_connections_user
                            FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """))
            _ensure_index(
                "dropbox_connections", "ix_dropbox_connections_user_id", "user_id"
            )
            _ensure_index(
                "dropbox_connections",
                "ix_dropbox_connections_remote_folder_id",
                "remote_folder_id",
            )
            logger.info("Migration: created table dropbox_connections")
        except Exception as exc:
            logger.warning("Migration skipped dropbox_connections: %s", exc)

    if "dropbox_document_tombstones" not in existing_tables:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"""
                    CREATE TABLE dropbox_document_tombstones (
                        id                CHAR(36)      PRIMARY KEY,
                        user_id           CHAR(36)      NOT NULL,
                        external_file_id  VARCHAR(255)  NOT NULL,
                        created_at        {timestamp_type},
                        CONSTRAINT uq_dropbox_tombstone_user_file
                            UNIQUE (user_id, external_file_id),
                        CONSTRAINT fk_dropbox_tombstones_user
                            FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """))
            _ensure_index(
                "dropbox_document_tombstones",
                "ix_dropbox_tombstones_user_id",
                "user_id",
            )
            _ensure_index(
                "dropbox_document_tombstones",
                "ix_dropbox_tombstones_external_file_id",
                "external_file_id",
            )
            logger.info("Migration: created table dropbox_document_tombstones")
        except Exception as exc:
            logger.warning("Migration skipped dropbox_document_tombstones: %s", exc)

    # ── Response feedback table ────────────────────────────────────────────
    existing_tables = set(inspector.get_table_names())
    if "response_feedback" not in existing_tables:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"""
                    CREATE TABLE response_feedback (
                        id                  CHAR(36)        PRIMARY KEY,
                        user_id             CHAR(36)        NOT NULL,
                        session_id          CHAR(36),
                        message_id          CHAR(36)        NOT NULL,
                        question_message_id CHAR(36),
                        question            {text_type}     NOT NULL,
                        response            {text_type}     NOT NULL,
                        feedback            VARCHAR(10)     NOT NULL,
                        reason              VARCHAR(64),
                        comment             {text_type},
                        created_at          {timestamp_type} NOT NULL,
                        updated_at          {timestamp_type},
                        CONSTRAINT uq_response_feedback_message UNIQUE (message_id),
                        CONSTRAINT fk_response_feedback_user
                            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                        CONSTRAINT fk_response_feedback_session
                            FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE,
                        CONSTRAINT fk_response_feedback_message
                            FOREIGN KEY (message_id) REFERENCES chat_messages (id) ON DELETE CASCADE,
                        CONSTRAINT fk_response_feedback_question
                            FOREIGN KEY (question_message_id) REFERENCES chat_messages (id) ON DELETE SET NULL
                    )
                """))
            _ensure_index(
                "response_feedback", "ix_response_feedback_user_id", "user_id"
            )
            _ensure_index(
                "response_feedback", "ix_response_feedback_session_id", "session_id"
            )
            _ensure_index(
                "response_feedback", "ix_response_feedback_feedback", "feedback"
            )
            _ensure_index(
                "response_feedback", "ix_response_feedback_created_at", "created_at"
            )
            logger.info("Migration: created table response_feedback")
        except Exception as exc:
            logger.warning("Migration skipped response_feedback: %s", exc)


def advisory_lock(lock_id: int, timeout_seconds: int = 30):
    """Context manager that acquires a cross-process advisory lock.

    * SQLite — no-op, because SQLite serializes writes anyway.
    * MySQL — ``GET_LOCK``/``RELEASE_LOCK`` on a dedicated connection, so
      concurrent Celery workers and the cleanup scheduler never overlap.
    * Other server dialects — transaction-scoped advisory lock.

    Usage::

        with advisory_lock(hash("cleanup_inactive") & 0x7FFFFFFF):
            ...
    """
    if is_sqlite:
        # SQLite serializes writes; no explicit lock needed.
        return _noop_contextmanager()

    if is_mysql:
        @contextmanager
        def _mysql_lock():
            # GET_LOCK names are server-wide and capped at 64 characters.
            lock_name = f"pdf_rag:{lock_id}"
            conn = engine.connect()
            try:
                acquired = conn.execute(
                    text("SELECT GET_LOCK(:name, :timeout)"),
                    {"name": lock_name, "timeout": timeout_seconds},
                ).scalar()
                if acquired != 1:
                    logger.warning(
                        "Advisory lock %s not acquired within %ss",
                        lock_name,
                        timeout_seconds,
                    )
                yield
            finally:
                try:
                    conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": lock_name})
                except Exception:
                    logger.debug("Failed to release advisory lock %s", lock_name, exc_info=True)
                conn.close()

        return _mysql_lock()

    @contextmanager
    def _xact_lock():
        with engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(:id)"), {"id": lock_id})
            yield

    return _xact_lock()


def _noop_contextmanager():
    from contextlib import contextmanager as _cm

    @_cm
    def _noop():
        yield

    return _noop()


def init_db():
    """Create all tables on startup and apply schema migrations."""
    from app import models  # noqa: F401 — import to register models
    Base.metadata.create_all(bind=engine)
    _migrate_schema()
    try:
        from app.services.prompt_management.seed import ensure_default_prompts

        with SessionLocal() as session:
            ensure_default_prompts(session)
            session.commit()
    except Exception as exc:
        logger.warning("Prompt seed skipped: %s", exc)
