"""
Application configuration via pydantic-settings.
All config is loaded from environment variables with sensible defaults.
A .env file is optional — local Ollama defaults work without one.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pydantic import model_validator, field_validator

class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "Genova_Document_Assistant"
    SECRET_KEY: str = "change-me-in-production-please"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:7860"
    # Comma-separated Host header allowlist for TrustedHostMiddleware.
    # Empty → localhost defaults in development; must be set in production.
    ALLOWED_HOSTS: str = ""

    # ── Logging ──────────────────────────────────────────
    LOG_LEVEL: str | None = None
    LOG_FILE: str = "./data/logs/app.log"


    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./data/app.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_PRE_PING: bool = True

    # ── Auth ─────────────────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_EXPIRY_MINUTES: int = 15
    JWT_REFRESH_EXPIRY_DAYS: int = 7
    # Auth bypass disabled for security
    DEV_AUTH_BYPASS: bool = False
    # Optional bearer token required to scrape /metrics in production
    METRICS_ACCESS_TOKEN: str = ""
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_DRIVE_REDIRECT_URI: str = "http://127.0.0.1:7860/api/v1/auth/google-drive/callback"
    # Dropbox document source (OAuth + public shared links). Credentials live in
    # the backend .env and are never exposed to the browser.
    DROPBOX_APP_KEY: str = ""
    DROPBOX_APP_SECRET: str = ""
    DROPBOX_REDIRECT_URI: str = "http://127.0.0.1:7860/api/v1/auth/dropbox/callback"
    # Preconfigured shared folder shown in Settings → Integrations for testing.
    DROPBOX_TEST_SHARED_LINK: str = (
        "https://www.dropbox.com/scl/fo/bc9imczz3bxl5ya8l7cil/"
        "AN5l1yDnWejG9jYIR_fD6A0?rlkey=rp14xvckn7zin1bw4axe50bjt"
        "&st=tzxfo2y1&e=1&dl=0"
    )
    FRONTEND_URL: str = "http://localhost:3000"
    EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS: int = 24
    # POC / early deploy: allow login immediately after register (no email link).
    # Set true once SMTP is configured and verification emails are required.
    REQUIRE_EMAIL_VERIFICATION: bool = False

    # ── Email verification ───────────────────────────────
    MAIL_USERNAME: str = ""
    MAIL_PASSWORD: str = ""
    MAIL_FROM: str = ""
    MAIL_SERVER: str = ""
    MAIL_PORT: int = 587
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False

    # Google Drive background sync
    DRIVE_SYNC_ENABLED: bool = False
    DRIVE_SYNC_INTERVAL_MINUTES: int = 60
    # Dropbox autosync (new/updated files, remote deletions)
    DROPBOX_SYNC_ENABLED: bool = True
    DROPBOX_SYNC_INTERVAL_MINUTES: int = 60
    GOOGLE_SERVICE_ACCOUNT_FILE: str = ""

    # Celery / Redis background processing
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_TRACK_STARTED: bool = True
    # Response/embedding cache (optional). Read via Settings so .env is honored.
    REDIS_URL: str | None = None
    # When set, injected into redis:// URLs that lack a password (compose/local).
    REDIS_PASSWORD: str = ""
    CACHE_TTL: int = 3600
    CACHE_LRU_MAX_SIZE: int = 128

    # ── Auth cookies (HttpOnly session; static-export compatible) ──
    # Cookies are always set on login/refresh. Frontends should prefer cookies
    # over localStorage. Body tokens remain for API clients / tests when enabled.
    AUTH_RETURN_BODY_TOKENS: bool = True
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"  # lax | strict | none
    COOKIE_DOMAIN: str = ""
    AUTH_COOKIE_CSRF: bool = True

    # ── Document Processing ──────────────────────────────
    DOC_PROCESSING_TIMEOUT_MINUTES: int = 30
    DOC_PROCESSING_MAX_RETRIES: int = 3
    DOC_PROCESSING_RETRY_DELAY_SECONDS: int = 30
    DOC_CLEANUP_MAX_AGE_DAYS: int = 90

    # ── File Upload ──────────────────────────────────────
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: set = {"pdf", "docx", "txt", "md"}
    ALLOWED_MIME_TYPES: dict = {
        ".pdf": ["application/pdf"],
        ".docx": [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
        ],
        ".txt": ["text/plain"],
        ".md": ["text/markdown"],

    }

    # ── AWS S3 (per-user documents + sticky notes) ──────
    # Layout: s3://<bucket>/{user_id}/pdfs/<file>
    #     and s3://<bucket>/{user_id}/sticky_notes/notes.json
    # Optional S3_BASE_PREFIX inserts a root folder before {user_id}/.
    # Leave S3_BUCKET / AWS_S3_BUCKET empty to keep everything on local disk only.
    # Credentials come from the AWS provider chain (env vars, shared credentials
    # file, IAM role) — never from this file.
    S3_BUCKET: str = ""
    AWS_S3_BUCKET: str = ""          # preferred alias; copied into S3_BUCKET when set
    S3_BASE_PREFIX: str = "NoteBase"  # s3://bucket/NoteBase/{user_id}/…
    S3_REGION: str = ""              # falls back to AWS_REGION/AWS_DEFAULT_REGION
    S3_ENDPOINT_URL: str = ""        # optional: MinIO / LocalStack endpoint
    S3_PDFS_PREFIX: str = "pdfs"
    S3_NOTES_PREFIX: str = "sticky_notes"
    S3_SERVER_SIDE_ENCRYPTION: str = ""  # set to "AES256" to enable SSE (KMS)

    # ── RAG Pipeline ─────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 20 # Fetch more candidates for reranking
    TOP_K_RERANK: int = 8 # Final number of chunks to return after reranking

    # ── Hierarchical semantic chunking ───────────────────
    # The chunk-generation strategy is versioned so reindexing can supersede
    # old vectors without deleting the original uploaded PDF.
    #   semantic_llm  → LLM structure analysis with deterministic fallback
    #   deterministic → heading/paragraph based hierarchy, no LLM call
    #   legacy        → previous layout/recursive splitter (compatibility)
    CHUNK_STRATEGY: str = "semantic_llm"
    SEMANTIC_CHUNKING_ENABLED: bool = True
    SEMANTIC_CHUNKING_PROVIDER: str = "bedrock"  # bedrock (Nova 2 Lite) | ollama
    # Blank → BEDROCK_MODEL_ID (us.amazon.nova-2-lite-v1:0).
    SEMANTIC_CHUNKING_MODEL_ID: str = ""
    SEMANTIC_CHUNKING_MAX_INPUT_TOKENS: int = 6000
    SEMANTIC_CHUNKING_MAX_OUTPUT_TOKENS: int = 4096
    # Titan Text Embeddings V2 accepts 8192 tokens; stay conservative so a
    # chunk plus metadata is always safely embeddable.
    SEMANTIC_CHUNKING_MAX_CHUNK_TOKENS: int = 1800
    SEMANTIC_CHUNKING_MIN_CHUNK_TOKENS: int = 80
    SEMANTIC_CHUNKING_RETRIES: int = 2
    SEMANTIC_CHUNKING_TIMEOUT_SECONDS: int = 120
    SEMANTIC_CHUNKING_MAX_BLOCKS_PER_CALL: int = 120
    SEMANTIC_CHUNKING_BLOCK_PREVIEW_CHARS: int = 220
    SEMANTIC_CHUNKING_VERSION: str = "semantic-llm-v1"
    SEMANTIC_FALLBACK_VERSION: str = "semantic-fallback-v1"

    # ── Layout-aware PDF / OCR ───────────────────────────
    # Native text is kept when its extraction quality is good; scanned or
    # corrupt pages are OCR'd page-by-page. Diagnostics contain no secrets
    # and are safe to retain for reprocessing/audit.
    PDF_LAYOUT_AWARE_ENABLED: bool = True
    PDF_LAYOUT_WRITE_DIAGNOSTICS: bool = True
    PDF_LAYOUT_REPORT_DIR: str = "./data/layout_reports"
    PDF_OCR_ENABLED: bool = True
    PDF_OCR_BACKEND: str = "auto"  # auto | tesseract | rapidocr | easyocr
    PDF_OCR_DPI: int = 300
    PDF_OCR_PSM: int = 3
    PDF_OCR_LANG: str = "eng"
    PDF_OCR_TIMEOUT_SECONDS: int = 120
    PDF_OCR_MIN_CONFIDENCE: float = 20.0
    PDF_OCR_MIN_CHARS: int = 80
    PDF_OCR_MIN_ALNUM_RATIO: float = 0.55
    PDF_OCR_MAX_GARBAGE_RATIO: float = 0.15
    PDF_OCR_IMAGE_COVERAGE_THRESHOLD: float = 0.30
    PDF_OCR_TESSDATA_DIR: str = ""  # blank → TESSDATA_PREFIX / system tessdata
    PDF_TABLE_MAX_CHARS: int = 4000
    PDF_TABLE_SPLIT_ROWS: int = 40

    # ── Structured tables (extraction → storage → interpretation) ─────────
    # Complete tables are persisted independently of chunk fragments so a hit
    # on one row can still attach the whole grid.
    TABLE_STORE_DIR: str = "./data/tables"
    TABLE_INTERPRETATION_ENABLED: bool = True
    TABLE_MAX_CANDIDATES: int = 3
    TABLE_CONTEXT_MAX_CHARS: int = 9000
    TABLE_INTERPRETATION_MAX_TOKENS: int = 700

    # ── Hybrid Search / RRF ───────────────────────────────
    USE_HYBRID_SEARCH: bool = True   # set to False to fall back to vector-only
    RRF_K: int = 60                  # RRF rank constant; 60 is the standard default
    # Pull neighboring chunks when a hit is a table or references Table N.
    RAG_EXPAND_ADJACENT_CHUNKS: bool = True
    RAG_ADJACENT_CHUNK_WINDOW: int = 1
    # Per-query RAG debug payload (redacted by default).
    RAG_DEBUG_ENABLED: bool = False
    RAG_DEBUG_REDACT: bool = True

    # ── Embeddings ───────────────────────────────────────
    # Localhost default: Ollama nomic-embed-text (768-d).
    # Production (ENVIRONMENT=production): forced to Bedrock Titan V2 (1024-d).
    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSION: int = 768

    # ── ChromaDB ─────────────────────────────────────────
    # Separate persist dirs per embedding family so indexes are never mixed.
    CHROMA_PERSIST_DIR: str = "./data/chroma_db_ollama_nomic"

    # ── LLM ──────────────────────────────────────────────
    # Localhost default: Ollama qwen2.5. Per-request ChatRequest.llm_provider
    # can override to bedrock in non-production. Production forces Bedrock Nova.
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "mistral:7b"
    LLM_MAX_NEW_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.3
    SUMMARY_MAX_TOKENS: int = 512

    # Amazon Bedrock Nova 2 Lite (US inference profile).
    # Auth: AWS_BEARER_TOKEN_BEDROCK when set, else IAM / default AWS credentials.
    BEDROCK_MODEL_ID: str = "us.amazon.nova-2-lite-v1:0"
    BEDROCK_REGION: str = "us-east-1"

    # ── Field-level Encryption ────────────────────────
    # Dedicated key for encrypting sensitive user fields (tokens, secrets).
    # Must be overridden in production — validate_production() enforces this.
    # Generate a strong key: python -c "import secrets; print(secrets.token_urlsafe(32))"
    FIELD_ENCRYPTION_KEY: str = "change-me-in-production-field-encryption-key"
    # Previous key kept during rotation so existing ciphertext remains readable.
    FIELD_ENCRYPTION_KEY_PREVIOUS: str = ""
    FIELD_ENCRYPTION_KEY_VERSION: int = 1

    # ── Document Cleanup ─────────────────────────────
    DOC_CLEANUP_ENABLED: bool = True
    DOC_CLEANUP_INACTIVE_DAYS: int = 30

    # ── LangSmith Tracing (optional) ─────────────────────
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "pdf-assistant-rag"

    # ── Reranker ─────────────────────────────────────────
    # Set RERANKER_ENABLED=true to enable BGE cross-encoder reranking.
    RERANKER_ENABLED: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    # ── Vision / Image captioning ─────────────────────
    VISION_PROVIDER: str | None = None  # e.g. 'openai'
    VISION_MODEL: str | None = None
    OPENAI_API_KEY: str = ""

    # ── Workspace Invitation ─────────────────────────
    APP_URL: str = "http://localhost:3000"
    INVITE_TOKEN_EXPIRY_HOURS: int = 72
    EMAIL_FROM: str = "no-reply@example.com"
    SMTP_HOST: str = ""
    SMTP_PORT: int = 0
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""

    # ── Database Tuning ─────────────────────────────────
    DATABASE_POOL_RECYCLE: int = 3600
    DATABASE_SLOW_QUERY_THRESHOLD: float = 1.0

    # ── Request Limits ───────────────────────────────────
    MAX_REQUEST_BODY_SIZE_MB: int = 50

    @property
    def cors_origins(self) -> list[str]:
        # Local dev defaults only. Production origins MUST come from ALLOWED_ORIGINS
        # — never hardcode abandoned deploy hosts (subdomain takeover risk).
        origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
        if self.ALLOWED_ORIGINS:
            for o in self.ALLOWED_ORIGINS.split(","):
                o_strip = o.strip()
                if o_strip and o_strip not in origins:
                    origins.append(o_strip)
        return origins

    @property
    def trusted_hosts(self) -> list[str]:
        """Hostnames accepted by TrustedHostMiddleware (FASTAPI-DEPLOY baseline)."""
        configured = [
            h.strip() for h in (self.ALLOWED_HOSTS or "").split(",") if h.strip()
        ]
        if configured:
            return configured
        if self.ENVIRONMENT == "production":
            # Fail closed-ish: only the configured frontend host + localhost probes.
            hosts = {"localhost", "127.0.0.1"}
            try:
                from urllib.parse import urlparse

                front = urlparse(self.FRONTEND_URL or "").hostname
                if front:
                    hosts.add(front)
            except Exception:
                pass
            return sorted(hosts)
        return ["localhost", "127.0.0.1", "testserver"]

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _normalize_database_url(cls, value: str) -> str:
        """Upgrade bare MySQL URLs to the bundled PyMySQL driver.

        ``mysql://`` and ``mariadb://`` are not usable by SQLAlchemy on their
        own; both are rewritten to ``mysql+pymysql://`` unless an explicit
        driver (e.g. ``mysql+mysqldb://``) was already supplied.
        """
        if not value:
            return value
        url = str(value).strip()
        if url.startswith("mysql://"):
            return "mysql+pymysql://" + url[len("mysql://"):]
        if url.startswith("mariadb://"):
            return "mysql+pymysql://" + url[len("mariadb://"):]
        return url

    @field_validator("LLM_PROVIDER", "EMBEDDING_PROVIDER", mode="before")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        if value is None:
            return value
        normalized = str(value).strip().lower()
        if normalized in {"bedrock_nova", "nova", "nova2", "nova_2_lite"}:
            return "bedrock"
        return normalized

    @model_validator(mode="after")
    def validate_vision_provider_keys(self) -> "Settings":
        provider = self.VISION_PROVIDER.lower() if self.VISION_PROVIDER else None
        if provider == "openai" and not self.OPENAI_API_KEY:
            raise ValueError(
                "ValidationError: OPENAI_API_KEY is required when VISION_PROVIDER is set to 'openai'."
            )
        return self

    @model_validator(mode="after")
    def resolve_s3_settings(self) -> "Settings":
        """Prefer AWS_S3_BUCKET when S3_BUCKET is empty; normalize optional base prefix."""
        bucket = (self.S3_BUCKET or self.AWS_S3_BUCKET or "").strip()
        if bucket:
            object.__setattr__(self, "S3_BUCKET", bucket)
            object.__setattr__(self, "AWS_S3_BUCKET", bucket)

        # Default layout: NoteBase/{user_id}/pdfs/... and .../sticky_notes/...
        prefix = (self.S3_BASE_PREFIX or "NoteBase").strip().strip("/")
        object.__setattr__(self, "S3_BASE_PREFIX", prefix or "NoteBase")
        return self

    @model_validator(mode="after")
    def apply_provider_defaults(self) -> "Settings":
        """Align providers with deployment environment and embedding family.

        - Localhost / development: Ollama LLM + nomic-embed-text
        - Production: Bedrock Nova LLM + Bedrock Titan embeddings
        Does not rewrite values the user explicitly set via environment variables,
        except that production never keeps Ollama as the active provider.
        """
        import os

        is_prod = (self.ENVIRONMENT or "").strip().lower() == "production"

        # Production is Bedrock-only for generation and embeddings.
        if is_prod:
            if self.LLM_PROVIDER == "ollama":
                object.__setattr__(self, "LLM_PROVIDER", "bedrock")
            if self.EMBEDDING_PROVIDER == "ollama":
                object.__setattr__(self, "EMBEDDING_PROVIDER", "bedrock")
            if "LLM_PROVIDER" not in os.environ and self.LLM_PROVIDER not in {
                "bedrock",
                "huggingface",
            }:
                object.__setattr__(self, "LLM_PROVIDER", "bedrock")
            if "EMBEDDING_PROVIDER" not in os.environ:
                object.__setattr__(self, "EMBEDDING_PROVIDER", "bedrock")

        if self.EMBEDDING_PROVIDER == "ollama":
            if "EMBEDDING_MODEL" not in os.environ and (
                not self.EMBEDDING_MODEL
                or self.EMBEDDING_MODEL.startswith("sentence-transformers/")
                or self.EMBEDDING_MODEL.startswith("amazon.")
            ):
                object.__setattr__(self, "EMBEDDING_MODEL", "nomic-embed-text")
            if "EMBEDDING_DIMENSION" not in os.environ and self.EMBEDDING_DIMENSION in {
                384,
                1024,
            }:
                object.__setattr__(self, "EMBEDDING_DIMENSION", 768)
            if (
                "CHROMA_PERSIST_DIR" not in os.environ
                and self.CHROMA_PERSIST_DIR
                in {
                    "./data/chroma_db",
                    "data/chroma_db",
                    "./data/chroma_db_bedrock_titan_v2",
                    "data/chroma_db_bedrock_titan_v2",
                }
            ):
                object.__setattr__(
                    self, "CHROMA_PERSIST_DIR", "./data/chroma_db_ollama_nomic"
                )

        if self.EMBEDDING_PROVIDER == "bedrock":
            if "EMBEDDING_MODEL" not in os.environ and (
                not self.EMBEDDING_MODEL
                or self.EMBEDDING_MODEL.startswith("sentence-transformers/")
                or self.EMBEDDING_MODEL == "nomic-embed-text"
            ):
                object.__setattr__(
                    self, "EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"
                )
            if "EMBEDDING_DIMENSION" not in os.environ and self.EMBEDDING_DIMENSION in {
                384,
                768,
            }:
                object.__setattr__(self, "EMBEDDING_DIMENSION", 1024)
            if (
                "CHROMA_PERSIST_DIR" not in os.environ
                and self.CHROMA_PERSIST_DIR
                in {
                    "./data/chroma_db",
                    "data/chroma_db",
                    "./data/chroma_db_ollama_nomic",
                    "data/chroma_db_ollama_nomic",
                }
            ):
                object.__setattr__(
                    self, "CHROMA_PERSIST_DIR", "./data/chroma_db_bedrock_titan_v2"
                )

        # LLM_MODEL is always the Ollama chat model (default mistral:7b).
        # Bedrock uses BEDROCK_MODEL_ID via llm_model_for_provider() so the
        # frontend can switch Ollama <-> Nova without poisoning Ollama's model id.
        if "LLM_MODEL" not in os.environ:
            model = (self.LLM_MODEL or "").strip()
            looks_like_bedrock = (
                model == (self.BEDROCK_MODEL_ID or "").strip()
                or model.startswith("us.")
                or model.startswith("amazon.")
                or "nova" in model.lower()
                or ("/" in model and not model.startswith("qwen"))
            )
            if not model or looks_like_bedrock:
                object.__setattr__(self, "LLM_MODEL", "mistral:7b")

        return self

    @property
    def is_production(self) -> bool:
        return (self.ENVIRONMENT or "").strip().lower() == "production"

    def available_llm_providers(self) -> list[str]:
        """Providers selectable in the UI / per-request overrides."""
        if self.is_production:
            return ["bedrock"]
        return ["ollama", "bedrock"]

    @property
    def default_llm_provider(self) -> str:
        available = self.available_llm_providers()
        if self.LLM_PROVIDER in available:
            return self.LLM_PROVIDER
        return available[0]

    @property
    def uses_ollama_llm(self) -> bool:
        return self.LLM_PROVIDER == "ollama"

    @property
    def uses_bedrock_llm(self) -> bool:
        return self.LLM_PROVIDER == "bedrock"

    @property
    def uses_direct_rag_llm(self) -> bool:
        """Retrieve-then-generate path (Ollama or Bedrock)."""
        return self.LLM_PROVIDER in {"ollama", "bedrock"}

    @property
    def uses_ollama_embeddings(self) -> bool:
        return self.EMBEDDING_PROVIDER == "ollama"

    @property
    def uses_bedrock_embeddings(self) -> bool:
        return self.EMBEDDING_PROVIDER == "bedrock"

    def validate_production(self) -> None:
        """Validate that critical secrets are overridden in production.

        Called during application startup (see ``main.py`` lifespan). In
        non-production environments it only logs warnings; in production it
        raises ``ValueError`` to prevent the app from starting with insecure
        defaults.
        """
        import logging
        _logger = logging.getLogger(__name__)

        _INSECURE_DEFAULTS = {
            "SECRET_KEY": "change-me-in-production-please",
            "FIELD_ENCRYPTION_KEY": "change-me-in-production-field-encryption-key",
        }

        issues: list[str] = []
        for field, default_value in _INSECURE_DEFAULTS.items():
            current = getattr(self, field, None)
            if current == default_value:
                issues.append(
                    f"{field} is set to the insecure default. "
                    f"Override it via environment variable or .env file."
                )

        # Minimum key length check
        if len(self.SECRET_KEY) < 32:
            issues.append(
                "SECRET_KEY is too short — use at least 32 characters. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        if len(self.FIELD_ENCRYPTION_KEY or "") < 32:
            issues.append(
                "FIELD_ENCRYPTION_KEY is too short — use at least 32 characters."
            )

        if self.DEV_AUTH_BYPASS:
            issues.append("DEV_AUTH_BYPASS must be false in production.")

        origins = [o.strip() for o in (self.ALLOWED_ORIGINS or "").split(",") if o.strip()]
        if not origins:
            issues.append(
                "ALLOWED_ORIGINS must list at least one explicit frontend origin in production."
            )
        if any(o == "*" for o in origins):
            issues.append(
                "ALLOWED_ORIGINS must not use '*' when credentials are enabled."
            )

        hosts = [h.strip() for h in (self.ALLOWED_HOSTS or "").split(",") if h.strip()]
        if not hosts:
            issues.append(
                "ALLOWED_HOSTS must be set explicitly in production "
                "(comma-separated API hostnames)."
            )
        if any(h == "*" for h in hosts):
            issues.append("ALLOWED_HOSTS must not contain '*'.")

        if (self.COOKIE_SAMESITE or "").lower() == "none" and not self.COOKIE_SECURE:
            # cookies.py forces Secure when SameSite=None; still flag misconfig.
            if self.ENVIRONMENT == "production":
                issues.append(
                    "COOKIE_SAMESITE=none requires COOKIE_SECURE=true in production."
                )

        if self.ENVIRONMENT == "production" and issues:
            raise ValueError(
                "Production configuration errors:\n  • " + "\n  • ".join(issues)
            )
        elif issues and self.ENVIRONMENT == "production":
            # unreachable; kept for clarity
            pass
        elif issues:
            # In non-production, only warn about insecure defaults / short keys —
            # do not require ALLOWED_HOSTS / ORIGINS for local DX.
            soft = [
                i
                for i in issues
                if "ALLOWED_ORIGINS" not in i
                and "ALLOWED_HOSTS" not in i
                and "DEV_AUTH_BYPASS" not in i
            ]
            for issue in soft:
                _logger.warning("Non-production config warning: %s", issue)

    # .env is optional — missing file is ignored; OS env vars still work.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — loaded once on startup."""
    return Settings()
