"""
SQLAlchemy ORM models for users, documents, and chat messages.
"""
import uuid
import enum
import base64
import hashlib
from datetime import datetime, timezone

from cryptography.fernet import Fernet
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

    def _get_cipher(self):
        from app.config import get_settings
        settings = get_settings()
        key = base64.urlsafe_b64encode(
            hashlib.sha256(settings.FIELD_ENCRYPTION_KEY.encode()).digest()
        )
        return Fernet(key)

    def process_bind_param(self, value, dialect):
        """Encrypt the value before saving to the database."""
        if value is None:
            return value
        cipher = self._get_cipher()
        return cipher.encrypt(value.encode()).decode()

    def process_result_value(self, value, dialect):
        """Decrypt the value after reading from the database."""
        if value is None:
            return value
        cipher = self._get_cipher()
        try:
            return cipher.decrypt(value.encode()).decode()
        except Exception:
            return value


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
    hf_token = Column(EncryptedString, nullable=True)
    display_name = Column(String(120), nullable=True)
    avatar_url = Column(String(500), nullable=True)

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
    workspace_memberships = relationship(
        "WorkspaceMember",
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
    workspace_name = Column(String(255), nullable=False)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(Timestamp, nullable=False)
    accepted_at = Column(Timestamp, nullable=True)

    inviter = relationship("User")


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

    # Relationships
    owner = relationship("User", back_populates="documents")
    messages = relationship(
        "ChatMessage",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class ChatMessage(Base):
    """
    Persistent log of conversations between users and the AI analyst.
    """
    __tablename__ = "chat_messages"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    document_id = Column(
        GUID,
        ForeignKey("documents.id"),
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


class DriveConnection(Base):
    __tablename__ = "drive_connections"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    folder_id = Column(String(255), nullable=False, index=True)
    credentials_json = Column(LongText, nullable=True)
    service_account_file = Column(String(500), nullable=True)
    enabled = Column(Boolean, default=True, nullable=False)
    last_synced_at = Column(Timestamp, nullable=True)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="drive_connections")


# ── Knowledge graph (GraphRAG) ────────────────────────────────────────────────
# The graph is scoped to a user (the user's document collection), so entities
# can be shared across every document they upload. Provenance is stored on
# mentions/evidence rows rather than on the entity itself, which keeps every
# statement traceable to the document, page and chunk it came from.


class GraphEntity(Base):
    """A canonical entity/concept shared across a user's documents."""

    __tablename__ = "graph_entities"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "entity_type",
            "normalized_name",
            name="uq_graph_entity_identity",
        ),
    )

    id = Column(String(64), primary_key=True)
    user_id = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    canonical_name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(64), nullable=False, index=True)
    description = Column(LongText, nullable=True)
    aliases_json = Column(LongText, nullable=True)
    mention_count = Column(Integer, default=0, nullable=False)
    document_count = Column(Integer, default=0, nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    # Set when a fuzzy candidate existed but merging was not safe.
    possible_duplicate_of = Column(String(64), nullable=True)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))


class GraphEntityMention(Base):
    """One piece of evidence that an entity appears in a document chunk."""

    __tablename__ = "graph_entity_mentions"

    id = Column(String(64), primary_key=True)
    entity_id = Column(
        String(64),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(GUID, nullable=False, index=True)
    document_id = Column(GUID, nullable=False, index=True)
    chunk_id = Column(String(128), nullable=True)
    page = Column(Integer, nullable=True)
    section = Column(String(255), nullable=True)
    evidence_text = Column(LongText, nullable=True)
    confidence = Column(Float, default=0.0, nullable=False)
    extraction_method = Column(String(32), default="llm", nullable=False)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))


class GraphRelationship(Base):
    """A directed, evidence-backed edge between two canonical entities."""

    __tablename__ = "graph_relationships"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "source_entity_id",
            "target_entity_id",
            "relation_type",
            name="uq_graph_relationship",
        ),
    )

    id = Column(String(64), primary_key=True)
    user_id = Column(GUID, nullable=False, index=True)
    source_entity_id = Column(
        String(64),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_entity_id = Column(
        String(64),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relation_type = Column(String(64), nullable=False, index=True)
    description = Column(LongText, nullable=True)
    confidence = Column(Float, default=0.0, nullable=False)
    extraction_method = Column(String(32), default="llm", nullable=False)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))


class GraphRelationshipEvidence(Base):
    """Supporting passage for a relationship, kept per source document."""

    __tablename__ = "graph_relationship_evidence"

    id = Column(String(64), primary_key=True)
    relationship_id = Column(
        String(64),
        ForeignKey("graph_relationships.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(GUID, nullable=False, index=True)
    document_id = Column(GUID, nullable=False, index=True)
    chunk_id = Column(String(128), nullable=True)
    page = Column(Integer, nullable=True)
    section = Column(String(255), nullable=True)
    evidence_text = Column(LongText, nullable=True)
    confidence = Column(Float, default=0.0, nullable=False)
    created_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))


class GraphIndexStatus(Base):
    """Per-document indexing state so failures are visible and retryable."""

    __tablename__ = "graph_index_status"

    document_id = Column(GUID, primary_key=True)
    user_id = Column(GUID, nullable=False, index=True)
    status = Column(String(20), default="pending", nullable=False)
    entity_count = Column(Integer, default=0, nullable=False)
    relationship_count = Column(Integer, default=0, nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    attempts = Column(Integer, default=0, nullable=False)
    error = Column(LongText, nullable=True)
    updated_at = Column(Timestamp, default=lambda: datetime.now(timezone.utc))
