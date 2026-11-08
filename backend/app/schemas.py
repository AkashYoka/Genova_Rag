"""
Pydantic schemas for API request/response validation.
"""
import json
import re
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Any, Dict
from datetime import datetime
from app.models import UserRole, WorkspaceRole
from app.password_validation import validate_password


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorEnvelope(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = {}
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorEnvelope


class ValidationErrorResponse(BaseModel):
    error: ErrorEnvelope


# ── Auth ─────────────────────────────────────────────

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=80)
    email: EmailStr
    password: str = Field(..., min_length=8)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        validate_password(value)
        return value


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class EmailVerificationRequest(BaseModel):
    email: EmailStr


class MessageResponse(BaseModel):
    message: str


class RegistrationResponse(MessageResponse):
    email: EmailStr
    verification_url: Optional[str] = None


class GoogleLoginRequest(BaseModel):
    id_token: str = Field(..., min_length=10)


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username:Optional[str] = None

class UserProfileUpdate(BaseModel):
    username: Optional[str] = None
    display_name: Optional[str] = None
class UserUpdateResponse(BaseModel):
    id: str
    username: str
    email: EmailStr

class UpdatePassword(BaseModel):
    password: str
    confirm_password: str
    current_password: Optional[str] = None

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        validate_password(value)
        return value

class UpdatePasswordResponse(BaseModel):
    id: str
    username: str
    email: EmailStr
    password_changed: bool = True

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        validate_password(value)
        return value    


class WorkspaceInviteRequest(BaseModel):
    email: EmailStr
    workspace_name: str = Field(..., min_length=1, max_length=100)
    message: Optional[str] = None


class WorkspaceInviteResponse(BaseModel):
    email: EmailStr
    workspace_name: str
    invite_link: str
    expires_in_hours: int


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class RefreshRequest(BaseModel):
    refresh_token: str


class GoogleDriveAuthUrlResponse(BaseModel):
    auth_url: str


class GoogleDriveStatusResponse(BaseModel):
    connected: bool


class DriveFolderCreateRequest(BaseModel):
    url: str = Field(..., min_length=5, description="Google Drive folder URL or ID")


class DriveFolderResponse(BaseModel):
    id: str
    folder_id: str
    folder_name: Optional[str] = None
    folder_url: Optional[str] = None
    access_mode: str = "public"  # public | oauth
    enabled: bool = True
    sync_status: str = "idle"  # idle | queued | processing | completed | failed
    task_id: Optional[str] = None
    last_error: Optional[str] = None
    last_synced_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    document_count: int = 0

    class Config:
        from_attributes = True


class DriveFolderListResponse(BaseModel):
    connected: bool
    public_access_available: bool = True
    folders: list[DriveFolderResponse]


class DriveSyncStatusResponse(BaseModel):
    """Lightweight poll payload for a single Drive folder sync."""

    connection_id: str
    folder_id: str
    folder_name: Optional[str] = None
    access_mode: str = "public"
    sync_status: str
    task_id: Optional[str] = None
    last_error: Optional[str] = None
    last_synced_at: Optional[datetime] = None
    document_count: int = 0


class DriveSyncAcceptedResponse(BaseModel):
    """Returned when a Drive sync job is accepted (HTTP 202)."""

    message: str = "Drive sync queued"
    sync_status: str = "queued"
    connection_id: Optional[str] = None
    task_id: Optional[str] = None
    task_ids: list[str] = Field(default_factory=list)
    connections: int = 0
    already_running: bool = False
    folders: list[DriveFolderResponse] = Field(default_factory=list)


class DriveSyncResponse(BaseModel):
    """Legacy sync summary shape (kept for Celery task result payloads)."""

    connections: int = 0
    discovered: int = 0
    ingested: int = 0
    updated: int = 0
    skipped: int = 0
    deleted: int = 0
    failed: int = 0
    error: Optional[str] = None
    results: list[dict] = Field(default_factory=list)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key_preview: str
    created_at: datetime

    class Config:
        from_attributes = True


class ApiKeyCreateResponse(BaseModel):
    id: str
    name: str
    key_preview: str
    created_at: datetime
    raw_key: str

    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: UserRole
    is_admin: bool
    is_verified: bool
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Documents ────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    original_name: str = "Untitled Document"
    file_size: int = 0
    page_count: int = 0
    chunk_count: int = 0
    status: str = "pending"
    error_message: Optional[str] = None
    uploaded_at: Optional[datetime] = None
    summary: Optional[str] = None
    task_id: Optional[str] = None
    keywords: Optional[List[str]] = []
    extracted_urls: Optional[List[str]] = []
    processing_progress: Optional[int] = None
    processing_stage: Optional[str] = None
    s3_key: Optional[str] = None
    storage_backend: Optional[str] = None

    @field_validator("id", mode="before")
    @classmethod
    def parse_id(cls, v):
        return str(v) if v is not None else ""

    @field_validator("keywords", mode="before")
    @classmethod
    def parse_keywords(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return []

    @field_validator("extracted_urls", mode="before")
    @classmethod
    def parse_extracted_urls(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return []


    class Config:
        from_attributes = True

class BatchUploadResponse(BaseModel):
    documents: List[DocumentResponse]
    task_ids: List[str]
    total: int
    failed: List[str] = []

class DocumentRename(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Document name cannot be empty")
        return stripped


class DocumentUpdate(BaseModel):
    """Schema for updating document metadata via PATCH. All fields are optional
    so that callers can send a partial update (e.g. only the name or only the
    summary) without having to include every field."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    summary: Optional[str] = Field(None, max_length=5000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            stripped = value.strip()
            if not stripped:
                raise ValueError("Document name cannot be empty")
            return stripped
        return value


class DocumentStatusResponse(BaseModel):
    id: str
    status: str
    page_count: int
    chunk_count: int
    error_message: Optional[str] = None
    processing_progress: Optional[int] = None
    processing_stage: Optional[str] = None
    retry_count: Optional[int] = None
    last_error_traceback: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    items: List[DocumentResponse] = []
    total: int = 0
    page: int = 1
    pages: int = 1
    total_pages: int = 1
    limit: int = 20
    query: Optional[str] = None


# Admin

class DiskUsageResponse(BaseModel):
    total_bytes: int
    used_bytes: int
    free_bytes: int
    usage_percent: float
    upload_dir_bytes: int


class AdminStatsResponse(BaseModel):
    total_users: int
    total_pdfs_uploaded: int
    total_documents: int
    total_messages: int
    average_query_response_time_ms: float
    query_count: int
    disk_space_usage: DiskUsageResponse
    users: List[UserResponse]


# ── Chat ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    # Optional per-request override: ollama | bedrock | huggingface (default = server LLM_PROVIDER)
    llm_provider: Optional[str] = Field(default=None, max_length=32)

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        """Strip control characters (null bytes, ANSI escapes, etc.) from user input."""
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        stripped = cleaned.strip()
        if not stripped:
            raise ValueError("Question cannot be empty after sanitization")
        return stripped

    @field_validator("llm_provider")
    @classmethod
    def normalize_llm_provider(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        normalized = v.strip().lower()
        if not normalized:
            return None
        aliases = {
            "bedrock_nova": "bedrock",
            "nova": "bedrock",
            "nova2": "bedrock",
            "nova_2_lite": "bedrock",
            "qwen": "ollama",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"ollama", "bedrock", "huggingface"}:
            raise ValueError("llm_provider must be ollama, bedrock, or huggingface")
        return normalized


class SourceChunk(BaseModel):
    text: str
    filename: str
    page: int
    score: float
    confidence: float
    bbox: Optional[str] = None


class QuizOption(BaseModel):
    id: str
    text: str


class QuizSourceRef(BaseModel):
    filename: Optional[str] = None
    page: Optional[int] = None


class QuizQuestion(BaseModel):
    id: str
    question: str
    options: List[QuizOption]
    correct_option_id: str
    explanation: str = ""
    source: Optional[QuizSourceRef] = None


class QuizState(BaseModel):
    current_index: int = 0
    answers: dict[str, str] = {}
    completed: bool = False


class QuizPayload(BaseModel):
    id: str
    mode: str
    title: str
    questions: List[QuizQuestion]
    state: QuizState = Field(default_factory=QuizState)


class QuizStateUpdateRequest(BaseModel):
    """Persist interactive quiz progress on an assistant message."""
    current_index: Optional[int] = Field(default=None, ge=0)
    answers: Optional[dict[str, str]] = None
    completed: Optional[bool] = None


class GraphContextInfo(BaseModel):
    """Compact summary of the graph evidence used to ground an answer."""

    entities: List[str] = []
    relationship_count: int = 0
    source_count: int = 0
    truncated: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = []
    document_id: Optional[str] = None
    quiz: Optional[QuizPayload] = None
    graph: Optional[GraphContextInfo] = None


class FeedbackRequest(BaseModel):
    feedback: Optional[str] = Field(None, pattern="^(up|down)?$")


class AttachedDocument(BaseModel):
    id: str
    filename: str


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: List[SourceChunk] = []
    attachments: List[AttachedDocument] = []
    feedback: Optional[str] = None
    created_at: datetime
    usage: Optional[dict] = None
    llm_provider: Optional[str] = None
    response_time_ms: Optional[int] = None
    quiz: Optional[QuizPayload] = None
    graph: Optional[GraphContextInfo] = None

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessageResponse]
    document_id: Optional[str] = None

# Chunk settings schema for optional chunk size and overlap parameters in document processing
class ChunkSettings(BaseModel):
    chunk_size: int = Field(default=1000, ge=100, le=2000)
    chunk_overlap: int = Field(default=200, ge=0)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info: Any) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

class UploadUrl(BaseModel):
    url: str

class FeedbackRequest(BaseModel):
    feedback: Optional[str] = Field(None, pattern="^(up|down)?$")


# ── Chat Session ──────────────────────────────────────

class ChatSessionCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)

class ChatSessionUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Workspaces ────────────────────────────────────────

class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class WorkspaceMemberResponse(BaseModel):
    id: str
    workspace_id: str
    user_id: str
    role: WorkspaceRole
    joined_at: datetime

    class Config:
        from_attributes = True


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    created_by: str
    created_at: datetime

    class Config:
        from_attributes = True


class WorkspaceDetailResponse(BaseModel):
    """Workspace detail including the full member list."""
    id: str
    name: str
    created_by: str
    created_at: datetime
    members: List[WorkspaceMemberResponse] = []

    class Config:
        from_attributes = True


class WorkspaceMemberAdd(BaseModel):
    user_id: str = Field(..., min_length=1)
    role: WorkspaceRole = WorkspaceRole.viewer


class WorkspaceMemberRoleUpdate(BaseModel):
    role: WorkspaceRole


# ── AI Usage ──────────────────────────────────────────

class UsagePeriodTotals(BaseModel):
    request_count: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class UsageProviderBreakdown(BaseModel):
    provider: str
    request_count: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class UsageModelBreakdown(BaseModel):
    provider: str
    model_id: Optional[str] = None
    request_count: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class UsageTypeBreakdown(BaseModel):
    usage_type: str
    request_count: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class UsageUserBreakdown(BaseModel):
    user_id: str
    request_count: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class UsageSummaryResponse(BaseModel):
    start: str
    end: str
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    today: UsagePeriodTotals
    month_to_date: UsagePeriodTotals
    by_provider: List[UsageProviderBreakdown] = []
    by_model: List[UsageModelBreakdown] = []
    by_usage_type: List[UsageTypeBreakdown] = []
    by_user: List[UsageUserBreakdown] = []
    cost_disclaimer: str


class UsageDailyPoint(BaseModel):
    date: str
    request_count: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class UsageDailyResponse(BaseModel):
    start: str
    end: str
    points: List[UsageDailyPoint] = []


class UsageEventResponse(BaseModel):
    id: str
    created_at: str
    provider: str
    model_id: Optional[str] = None
    usage_type: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    cost_type: str
    pricing_note: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    document_id: Optional[str] = None


class UsageEventsResponse(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    items: List[UsageEventResponse] = []


class AdminDocumentItem(BaseModel):
    id: str
    user_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    original_name: str
    status: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None
    processing_stage: Optional[str] = None
    retry_count: Optional[int] = None
    uploaded_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AdminDocumentsResponse(BaseModel):
    total: int
    status_counts: dict = {}
    items: List[AdminDocumentItem] = []


class AdminDriveConnectionItem(BaseModel):
    id: str
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    folder_id: str
    folder_name: Optional[str] = None
    access_mode: Optional[str] = None
    enabled: bool = True
    sync_status: str
    last_error: Optional[str] = None
    last_synced_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AdminDriveConnectionsResponse(BaseModel):
    total: int
    items: List[AdminDriveConnectionItem] = []


class AdminWorkspaceItem(BaseModel):
    id: str
    name: str
    created_by: str
    creator_username: Optional[str] = None
    member_count: int = 0
    created_at: Optional[datetime] = None


class AdminWorkspacesResponse(BaseModel):
    total: int
    items: List[AdminWorkspaceItem] = []


class AdminApiKeyUserSummary(BaseModel):
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    total_keys: int = 0
    active_keys: int = 0
    last_used_at: Optional[datetime] = None


class AdminApiKeySummaryResponse(BaseModel):
    total_keys: int = 0
    active_keys: int = 0
    by_user: List[AdminApiKeyUserSummary] = []


# ── Dynamic Prompt Management ─────────────────────────

class PromptDomainCreate(BaseModel):
    slug: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = None
    is_active: bool = True
    is_fallback: bool = False
    sort_order: int = 100


class PromptDomainUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    description: Optional[str] = None
    is_active: Optional[bool] = None
    is_fallback: Optional[bool] = None
    sort_order: Optional[int] = None


class PromptDomainResponse(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    is_active: bool
    is_builtin: bool
    is_fallback: bool
    sort_order: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PromptTemplateCreate(BaseModel):
    domain_id: Optional[str] = None
    prompt_type: str = Field(
        ...,
        pattern=r"^(system|rag|greeting|agent|summarization|comparison|extraction|explanation|qa)$",
    )
    name: str = Field(..., min_length=1, max_length=160)
    content: str = Field(..., min_length=1)
    description: Optional[str] = None
    is_active: bool = True
    is_default: bool = False


class PromptTemplateUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=160)
    content: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None
    change_note: Optional[str] = Field(None, max_length=500)


class PromptTemplateResponse(BaseModel):
    id: str
    domain_id: Optional[str] = None
    domain_slug: Optional[str] = None
    prompt_type: str
    name: str
    content: str
    description: Optional[str] = None
    version: int
    is_active: bool
    is_default: bool
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PromptVersionResponse(BaseModel):
    id: str
    template_id: str
    version_number: int
    content: str
    change_note: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PromptPreviewRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = "Sample document excerpt for preview."
    document_ids: List[str] = []
    prompt_type: str = "rag"
    domain_slug: Optional[str] = None


class PromptPreviewResponse(BaseModel):
    domain: str
    confidence: float
    was_fallback: bool
    system_prompt: str
    task_prompt: str
    assembled_user_prompt: str
    rationale: Optional[str] = None


class DomainDetectRequest(BaseModel):
    question: str = Field(..., min_length=1)
    document_ids: List[str] = []
    context: Optional[str] = None


class DomainDetectResponse(BaseModel):
    domain: str
    confidence: float
    was_fallback: bool
    secondary_domains: List[str] = []
    rationale: Optional[str] = None
    cache_hit: bool = False


class PromptStatsResponse(BaseModel):
    domains: List[Dict[str, Any]] = []
    detections_total: int = 0
    fallback_total: int = 0
    low_confidence_total: int = 0
    queries_by_domain: Dict[str, int] = {}
    avg_confidence_by_domain: Dict[str, float] = {}
    recent_detections: List[Dict[str, Any]] = []
    recent_executions: List[Dict[str, Any]] = []


class DomainCardResponse(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    purpose: Optional[str] = None
    is_active: bool
    is_builtin: bool
    is_fallback: bool
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    version: Optional[int] = None
    is_prompt_active: bool = False
    last_updated: Optional[datetime] = None
    query_count: int = 0
    avg_confidence: Optional[float] = None


class PromptOverviewResponse(BaseModel):
    total_domains: int = 0
    active_domains: int = 0
    active_prompts: int = 0
    default_prompts: int = 0
    detections_total: int = 0
    fallback_total: int = 0
    last_updated: Optional[datetime] = None
    domains: List[DomainCardResponse] = []
    recent_detections: List[Dict[str, Any]] = []


class PromptTestRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = Field(
        default=(
            "Sample document excerpt used for admin testing only. "
            "Replace with real excerpts when validating production prompts."
        ),
        min_length=1,
    )
    domain_slug: Optional[str] = None
    auto_detect: bool = True
    prompt_type: str = "rag"
    document_ids: List[str] = []


class PromptTestResponse(BaseModel):
    domain: str
    confidence: float
    was_fallback: bool
    prompt_version: Optional[int] = None
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    system_prompt_preview: str
    answer: str
    rationale: Optional[str] = None


# Rebuild models for forward references
TokenResponse.model_rebuild()
