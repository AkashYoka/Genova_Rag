"""
SQLAlchemy ORM models for users, documents, and chat messages.
"""
import uuid
import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Boolean,
    Enum as SQLAlchemyEnum,
    Index,
    UniqueConstraint,
    text,
)
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.mysql import DATETIME as MYSQL_DATETIME, LONGTEXT as MYSQL_LONGTEXT
from sqlalchemy.orm import relationship

# ── Portable large-text column ────────────────────────────────────────────────
# MySQL's TEXT tops out at 64 KB, which is too small for full chat answers,
# extracted URL lists or tracebacks.  LONGTEXT keeps the historical "unbounded
# text" behaviour the application relies on; other dialects stay on TEXT.
LongText = Text().with_variant(MYSQL_LONGTEXT(), "mysql")

# ── Portable high-resolution timestamp ────────────────────────────────────────
# MySQL's plain DATETIME truncates fractional seconds, which would make
# messages created in the same second sort ambiguously.  DATETIME(6) keeps the
# microsecond precision the application has always stored.
Timestamp = DateTime().with_variant(MYSQL_DATETIME(fsp=6), "mysql")

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


# ── Document source / lifecycle constants ────────────────────────────────────
# Documents keep a single lifecycle column so every connector (uploads,
# Dropbox OAuth folders, Dropbox shared links) shares one ingestion pipeline.
DOCUMENT_SOURCE_UPLOAD = "upload"
DOCUMENT_SOURCE_DROPBOX = "dropbox"
DOCUMENT_SOURCE_DROPBOX_LINK = "dropbox_link"
DROPBOX_SOURCE_TYPES = (DOCUMENT_SOURCE_DROPBOX, DOCUMENT_SOURCE_DROPBOX_LINK)

LIFECYCLE_ACTIVE = "ACTIVE"
# User removed the document in Genova — autosync must never restore it.
LIFECYCLE_USER_DELETED = "USER_DELETED"
# The file disappeared from the external source — autosync may restore it.
LIFECYCLE_SOURCE_DELETED = "SOURCE_DELETED"


class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Stored as ``CHAR(36)`` on MySQL (and SQLite), which keeps every existing
    value readable as a plain lowercase UUID string.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return str(value)
        try:
            return str(uuid.UUID(value))
        except ValueError:
            return value

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return str(value)


class EncryptedString(TypeDecorator):
    """
    A custom SQLAlchemy type that transparently encrypts strings
    in the database using Fernet (AES). This ensures sensitive tokens
    aren't stored in plain text while remaining easily accessible in code.
    """
    impl = LongText
    cache_ok = False

    def process_bind_param(self, value, dialect):
        """Encrypt the value before saving to the database (primary key)."""
        if value is None:
            return value
        from app.security.encryption import encrypt_field

        return encrypt_field(value)

    def process_result_value(self, value, dialect):
        """Decrypt with primary then previous key (rotation-safe)."""
        if value is None:
            return value
        from app.security.encryption import decrypt_field

        plain = decrypt_field(value)
        # If neither key works, return ciphertext rather than crashing reads.
        return plain if plain is not None else value


class UserRole(str, enum.Enum):
    """
    Defines the available user roles for Role-Based Access Control (RBAC).
    - 'admin': Full access to system statistics and user management.
    - 'user': Standard access for uploading and chatting with documents.
    """
    user = "user"
    admin = "admin"


class WorkspaceRole(str, enum.Enum):
    admin = "admin"
    editor = "editor"
    viewer = "viewer"


class User(Base):
    """
    Represents a registered user within the system.
    Supports both legacy 'is_admin' flags and the modern 'role' enum for
    permissions.
    """
    __tablename__ = "users"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    google_refresh_token = Column(EncryptedString, nullable=True)
    # Dropbox OAuth (encrypted at rest, never sent to the frontend).
    dropbox_refresh_token = Column(EncryptedString, nullable=True)
    dropbox_access_token = Column(EncryptedString, nullable=True)
    dropbox_token_expires_at = Column(Timestamp, nullable=True)
    dropbox_account_id = Column(String(255), nullable=True, index=True)
    dropbox_display_name = Column(String(255), nullable=True)

    role = Column(
        SQLAlchemyEnum(UserRole),
        default=UserRole.user,
        nullable=False,
        server_default="user",
    )
    is_admin = Column(Boolean, default=False)
    # MySQL rejects a quoted 'true' default for BOOL; TRUE is valid on MySQL,
    # SQLite and PostgreSQL alike.
    is_verified = Column(Boolean, default=True, nullable=False, server_default=text("TRUE"))
    verification_token_hash = Column(String(64), nullable=True, index=True)
    verification_token_created_at = Column(Timestamp, nullable=True)

    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    last_login = Column(Timestamp, nullable=True, index=True)
    display_name = Column(String(120), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    # Bumped on logout / password change to invalidate outstanding JWTs
    token_version = Column(Integer, default=0, nullable=False, server_default="0")

    # Relationships
    documents = relationship(
        "Document",
        back_populates="owner",
        cascade="all, delete-orphan",
    )
    messages = relationship(
        "ChatMessage",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    api_keys = relationship(
        "ApiKey",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    chat_sessions = relationship(
        "ChatSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    drive_connections = relationship(
        "DriveConnection",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    dropbox_connections = relationship(
        "DropboxConnection",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    workspace_memberships = relationship(
        "WorkspaceMember",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    llm_usage = relationship(
        "LLMUsage",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class ApiKey(Base):
    """
    Stores secure hashes of API keys used for programmatic interaction with the
    system.
    """
    __tablename__ = "api_keys"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False, default="default")
    key_prefix = Column(String(20), nullable=False)
    hashed_key = Column(String(255), nullable=False, unique=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    last_used_at = Column(Timestamp, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")


class WorkspaceInvitation(Base):
    __tablename__ = "workspace_invitations"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    email = Column(String(120), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, unique=True, index=True)
    inviter_id = Column(
        GUID,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )
    # Canonical workspace identity — never resolve invitations by display name.
    workspace_id = Column(
        GUID,
        ForeignKey("workspaces.id"),
        nullable=True,
        index=True,
    )
    # Retained for email copy / legacy rows; not used for membership resolution.
    workspace_name = Column(String(255), nullable=False)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(Timestamp, nullable=False)
    accepted_at = Column(Timestamp, nullable=True)

    inviter = relationship("User")
    workspace = relationship("Workspace")


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)

    name = Column(String(255), nullable=False)

    created_by = Column(
        GUID,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    created_at = Column(
        Timestamp,
        default=lambda: datetime.now(timezone.utc),
    )

    creator = relationship("User")

    members = relationship(
        "WorkspaceMember",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"

    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "user_id",
            name="uq_workspace_member",
        ),
    )

    id = Column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )

    workspace_id = Column(
        GUID,
        ForeignKey("workspaces.id"),
        nullable=False,
        index=True,
    )

    user_id = Column(
        GUID,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    role = Column(
        SQLAlchemyEnum(WorkspaceRole),
        nullable=False,
        default=WorkspaceRole.viewer,
        server_default="viewer",
    )

    joined_at = Column(
        Timestamp,
        default=lambda: datetime.now(timezone.utc),
    )

    workspace = relationship(
        "Workspace",
        back_populates="members",
    )

    user = relationship(
        "User",
        back_populates="workspace_memberships",
    )


class ChatSession(Base):
    """
    Groups chat messages into logical sessions/threads.
    """
    __tablename__ = "chat_sessions"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class Document(Base):
    """
    Metadata and processing status for files uploaded by users.
    """
    __tablename__ = "documents"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False, index=True)
    original_name = Column(String(255), nullable=False)
    # Full S3 object key when stored in AWS (e.g. NoteBase/{user}/pdfs/{filename}).
    s3_key = Column(String(512), nullable=True, index=True)
    # "s3" for NoteBase uploads; "local" only for legacy pre-S3 documents.
    storage_backend = Column(String(20), nullable=True, default="local")
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="pending")
    error_message = Column(LongText, nullable=True)
    uploaded_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    last_accessed_at = Column(
        Timestamp,
        default=lambda: datetime.now(timezone.utc),
        nullable=True,
    )
    summary = Column(LongText, nullable=True)
    chunk_size = Column(Integer, nullable=True)
    chunk_overlap = Column(Integer, nullable=True)
    drive_file_id = Column(String(255), unique=True, nullable=True, index=True)
    drive_folder_id = Column(String(255), nullable=True, index=True)
    drive_synced_at = Column(Timestamp, nullable=True)
    # ISO-8601 modifiedTime from Drive API — used to detect remote updates.
    drive_modified_time = Column(String(64), nullable=True)
    # Generic external-source tracking (Dropbox and future connectors).
    # ``external_file_id`` is the provider's stable file ID (never the filename).
    source_type = Column(
        String(32),
        default=DOCUMENT_SOURCE_UPLOAD,
        nullable=False,
        index=True,
    )
    external_file_id = Column(String(255), nullable=True, index=True)
    external_folder_id = Column(String(255), nullable=True, index=True)
    external_revision = Column(String(128), nullable=True)
    external_synced_at = Column(Timestamp, nullable=True)
    lifecycle_status = Column(
        String(20),
        default=LIFECYCLE_ACTIVE,
        nullable=False,
        index=True,
    )
    source_deleted_at = Column(Timestamp, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at = Column(Timestamp, nullable=True)
    processing_progress = Column(Integer, default=0)
    processing_stage = Column(String(20), default="queued")
    retry_count = Column(Integer, default=0)
    last_error_traceback = Column(LongText, nullable=True)
    processing_started_at = Column(Timestamp, nullable=True)
    completed_at = Column(Timestamp, nullable=True)
    extracted_urls = Column(LongText, nullable=True)
    keywords = Column(LongText, nullable=True)
    # Auto-detected domain slug (medical, payment, education, …); null until classified.
    detected_domain = Column(String(64), nullable=True, index=True)
    domain_confidence = Column(Float, nullable=True)
    # ── Index versioning (hierarchical semantic chunking) ─────────────────
    # ``chunk_strategy`` records which generator produced the live index.
    # ``index_version`` is only advanced after a successful rebuild, so a
    # failed reindex never claims to have superseded older vectors.
    chunk_strategy = Column(String(32), nullable=True, default="semantic_llm")
    index_version = Column(String(32), nullable=True)
    # JSON validation/coverage report for the last successful indexing run.
    chunking_report = Column(LongText, nullable=True)

    # Relationships
    owner = relationship("User", back_populates="documents")
    # Chat history outlives the PDF: deleting a document must never cascade into
    # sessions/messages. document_id is SET NULL on hard delete.
    messages = relationship(
        "ChatMessage",
        back_populates="document",
        passive_deletes=True,
    )


# ── Hierarchical semantic chunk index ────────────────────────────────────────
# Sections and chunks are persisted in the relational database so the frontend
# can render an expandable document tree without scanning Chroma metadata.
# Chroma/BM25 remain the retrieval stores; these rows are the navigation index.


class DocumentSection(Base):
    """One node in the document hierarchy (Document → Section → Subsection)."""

    __tablename__ = "document_sections"
    __table_args__ = (
        Index(
            "ix_document_sections_document_ordinal",
            "document_id",
            "ordinal",
        ),
    )

    # Deterministic id derived from document id + hierarchy path.
    id = Column(String(64), primary_key=True)
    document_id = Column(
        GUID,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(GUID, nullable=False, index=True)
    parent_section_id = Column(String(64), nullable=True, index=True)
    title = Column(String(512), nullable=False)
    level = Column(Integer, nullable=False, default=1)
    ordinal = Column(Integer, nullable=False, default=0)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    summary = Column(LongText, nullable=True)
    block_ids_json = Column(LongText, nullable=True)
    # Associated tables: JSON list of {table_id, page, headers, caption}.
    tables_json = Column(LongText, nullable=True)
    chunk_count = Column(Integer, nullable=False, default=0)
    index_version = Column(String(32), nullable=True)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))


class DocumentChunk(Base):
    """Persisted semantic chunk with its original extracted content.

    The same content is embedded into Chroma and BM25. This row is the durable
    hierarchy record and keeps source block ids for audit/reconstruction.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index(
            "ix_document_chunks_document_index",
            "document_id",
            "chunk_index",
        ),
    )

    # Matches the Chroma/BM25 id: ``{document_id}_{chunk_index}``.
    id = Column(String(160), primary_key=True)
    document_id = Column(
        GUID,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(GUID, nullable=False, index=True)
    section_id = Column(
        String(64),
        ForeignKey("document_sections.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    parent_section_id = Column(String(64), nullable=True, index=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    content = Column(LongText, nullable=False)
    chunk_type = Column(String(32), nullable=False, default="text")
    content_type = Column(String(32), nullable=True)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    block_ids_json = Column(LongText, nullable=True)
    table_ids_json = Column(LongText, nullable=True)
    metadata_json = Column(LongText, nullable=True)
    index_version = Column(String(32), nullable=True)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))


class ChatMessage(Base):
    """
    Persistent log of conversations between users and the AI analyst.
    """
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index(
            "ix_chat_messages_session_user_created",
            "session_id",
            "user_id",
            "created_at",
        ),
    )

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    document_id = Column(
        GUID,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    session_id = Column(
        GUID,
        ForeignKey("chat_sessions.id"),
        nullable=True,
        index=True,
    )
    role = Column(String(20), nullable=False)  # "user" | "assistant"
    content = Column(LongText, nullable=False)
    sources_json = Column(LongText, nullable=True)
    feedback = Column(String(10), nullable=True)  # "up" | "down" | None
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="messages")
    document = relationship("Document", back_populates="messages")
    session = relationship("ChatSession", back_populates="messages")


class ResponseFeedback(Base):
    """A user's rating of one assistant response, with full context.

    One row per rated response. ``question`` and ``response`` are snapshots so
    an admin can review exactly what the user saw even if history is edited
    or deleted later; the message/session ids keep it traceable.
    """

    __tablename__ = "response_feedback"
    __table_args__ = (
        UniqueConstraint(
            "message_id",
            name="uq_response_feedback_message",
        ),
        Index(
            "ix_response_feedback_created_feedback",
            "created_at",
            "feedback",
        ),
    )

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(
        GUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id = Column(
        GUID,
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    message_id = Column(
        GUID,
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_message_id = Column(
        GUID,
        ForeignKey("chat_messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    question = Column(LongText, nullable=False)
    response = Column(LongText, nullable=False)
    feedback = Column(String(10), nullable=False, index=True)  # "up" | "down"
    reason = Column(String(64), nullable=True)
    comment = Column(LongText, nullable=True)
    created_at = Column(
        Timestamp,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    updated_at = Column(
        Timestamp,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class LLMUsage(Base):
    """Per-request AI usage ledger for authenticated users.

    Token fields are nullable when the provider does not report them.
    ``estimated_cost_usd`` is a calculated estimate from the pricing registry
    (or provider-reported when available) — never an invented figure.
    """

    __tablename__ = "llm_usage"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(
        GUID,
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    message_id = Column(
        GUID,
        ForeignKey("chat_messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    document_id = Column(
        GUID,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    provider = Column(String(32), nullable=False, index=True)
    model_id = Column(String(128), nullable=True, index=True)
    usage_type = Column(String(64), nullable=False, default="chat", index=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cache_read_tokens = Column(Integer, nullable=True)
    cache_write_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    estimated_cost_usd = Column(Float, nullable=True)
    # estimated | provider_reported | local_zero | unavailable
    cost_type = Column(String(32), nullable=False, default="unavailable")
    pricing_note = Column(String(512), nullable=True)
    created_at = Column(
        Timestamp,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    user = relationship("User", back_populates="llm_usage")


class DriveConnection(Base):
    """A Google Drive folder the user attached for PDF sync into RAG."""

    __tablename__ = "drive_connections"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "folder_id",
            name="uq_drive_connection_user_folder",
        ),
    )

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    folder_id = Column(String(255), nullable=False, index=True)
    folder_name = Column(String(255), nullable=True)
    folder_url = Column(String(500), nullable=True)
    # public = Anyone-with-link (no user OAuth); oauth = private via user token
    access_mode = Column(String(20), default="public", nullable=False)
    # Legacy fields — user OAuth (User.google_refresh_token) is preferred.
    credentials_json = Column(LongText, nullable=True)
    service_account_file = Column(String(500), nullable=True)
    enabled = Column(Boolean, default=True, nullable=False)
    # idle | queued | processing | completed | failed
    # (legacy aliases still seen in older rows: syncing≈processing, ok≈completed, error≈failed)
    sync_status = Column(String(20), default="idle", nullable=False)
    celery_task_id = Column(String(255), nullable=True)
    last_error = Column(LongText, nullable=True)
    last_synced_at = Column(Timestamp, nullable=True)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="drive_connections")


class DropboxConnection(Base):
    """A Dropbox folder (OAuth) or shared link attached for PDF sync into RAG."""

    __tablename__ = "dropbox_connections"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "connection_kind",
            "remote_folder_id",
            name="uq_dropbox_connection_user_folder",
        ),
    )

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    # oauth = private folder in the user's account; link = public shared link
    connection_kind = Column(String(20), default="oauth", nullable=False)
    # Dropbox folder ID/path for OAuth, or a stable key derived from the link.
    remote_folder_id = Column(String(255), nullable=False, index=True)
    # Path inside the Dropbox account (OAuth) or inside the shared link.
    folder_path = Column(String(1024), nullable=True)
    shared_link = Column(String(2048), nullable=True)
    folder_name = Column(String(255), nullable=True)
    folder_url = Column(String(2048), nullable=True)
    # Initial import selection (JSON list of Dropbox file IDs); cleared on sync.
    pending_file_ids = Column(LongText, nullable=True)
    enabled = Column(Boolean, default=True, nullable=False)
    # idle | queued | processing | completed | failed
    sync_status = Column(String(20), default="idle", nullable=False)
    celery_task_id = Column(String(255), nullable=True)
    last_error = Column(LongText, nullable=True)
    last_synced_at = Column(Timestamp, nullable=True)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="dropbox_connections")


class DropboxDocumentTombstone(Base):
    """Remembers user-deleted Dropbox files so autosync cannot re-import them.

    The tombstone outlives the Document row (trash purge / hard delete), which
    keeps deletion intent intact until the user manually re-imports the file.
    """

    __tablename__ = "dropbox_document_tombstones"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "external_file_id",
            name="uq_dropbox_tombstone_user_file",
        ),
    )

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    external_file_id = Column(String(255), nullable=False, index=True)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))


# ── Dynamic Prompt Management ─────────────────────────────────────────────────


class PromptDomain(Base):
    """Configurable domain used for automatic prompt selection."""

    __tablename__ = "prompt_domains"

    id = Column(GUID, primary_key=True, default=generate_uuid)
    slug = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(120), nullable=False)
    description = Column(LongText, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_builtin = Column(Boolean, default=False, nullable=False)
    is_fallback = Column(Boolean, default=False, nullable=False)
    sort_order = Column(Integer, default=100, nullable=False)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        Timestamp,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    templates = relationship(
        "PromptTemplate",
        back_populates="domain",
        cascade="all, delete-orphan",
    )


class PromptTemplate(Base):
    """Layered prompt content: global base, domain overlay, or task template."""

    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint(
            "domain_id",
            "prompt_type",
            "name",
            name="uq_prompt_templates_domain_type_name",
        ),
    )

    id = Column(GUID, primary_key=True, default=generate_uuid)
    # Null domain_id = global layer (base / shared task defaults).
    domain_id = Column(
        GUID,
        ForeignKey("prompt_domains.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    # system | rag | greeting | agent | summarization | comparison | extraction | explanation | qa
    prompt_type = Column(String(40), nullable=False, index=True)
    name = Column(String(160), nullable=False)
    content = Column(LongText, nullable=False)
    description = Column(LongText, nullable=True)
    version = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_default = Column(Boolean, default=False, nullable=False)
    created_by = Column(GUID, ForeignKey("users.id"), nullable=True)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        Timestamp,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    domain = relationship("PromptDomain", back_populates="templates")
    versions = relationship(
        "PromptVersion",
        back_populates="template",
        cascade="all, delete-orphan",
        order_by="PromptVersion.version_number.desc()",
    )


class PromptVersion(Base):
    """Immutable snapshot of a prompt template for restore / audit."""

    __tablename__ = "prompt_versions"
    __table_args__ = (
        UniqueConstraint(
            "template_id",
            "version_number",
            name="uq_prompt_versions_template_version",
        ),
    )

    id = Column(GUID, primary_key=True, default=generate_uuid)
    template_id = Column(
        GUID,
        ForeignKey("prompt_templates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_number = Column(Integer, nullable=False)
    content = Column(LongText, nullable=False)
    change_note = Column(String(500), nullable=True)
    created_by = Column(GUID, ForeignKey("users.id"), nullable=True)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))

    template = relationship("PromptTemplate", back_populates="versions")


class DomainDetectionLog(Base):
    """Trace of automatic domain classification decisions."""

    __tablename__ = "domain_detection_logs"

    id = Column(GUID, primary_key=True, default=generate_uuid)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(GUID, nullable=True, index=True)
    document_ids_json = Column(LongText, nullable=True)
    question_preview = Column(String(500), nullable=True)
    detected_domain = Column(String(64), nullable=False, index=True)
    secondary_domains_json = Column(LongText, nullable=True)
    confidence = Column(Float, default=0.0, nullable=False)
    was_fallback = Column(Boolean, default=False, nullable=False)
    rationale = Column(LongText, nullable=True)
    cache_hit = Column(Boolean, default=False, nullable=False)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc), index=True)


class PromptExecutionLog(Base):
    """Which prompts were assembled for a chat turn."""

    __tablename__ = "prompt_execution_logs"

    id = Column(GUID, primary_key=True, default=generate_uuid)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(GUID, nullable=True, index=True)
    domain_slug = Column(String(64), nullable=False, index=True)
    confidence = Column(Float, default=0.0, nullable=False)
    prompt_type = Column(String(40), nullable=False)
    system_template_id = Column(GUID, nullable=True)
    system_version = Column(Integer, nullable=True)
    task_template_id = Column(GUID, nullable=True)
    task_version = Column(Integer, nullable=True)
    domain_template_id = Column(GUID, nullable=True)
    domain_version = Column(Integer, nullable=True)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc), index=True)
