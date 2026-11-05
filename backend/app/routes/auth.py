"""
Auth API routes — register, login, and user profile.
"""
import re
import secrets
import logging
import hashlib
from html import escape
from typing import Optional, List
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, Query, Request, status, Response, Body
from fastapi.responses import HTMLResponse
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
from google_auth_oauthlib.flow import Flow
from langsmith import expect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.config import get_settings
from app.database import get_db
from app.exceptions import (
    AppException,
    ConflictException,
    ExternalServiceException,
    NotFoundException,
    UnauthorizedException,
    ValidationException,
)
from app.models import User, ApiKey, UserRole
from app.schemas import (
    GoogleLoginRequest,
    GoogleDriveAuthUrlResponse,
    GoogleDriveStatusResponse,
    RefreshRequest,
    TokenResponse,
    UpdatePassword,
    UpdatePasswordResponse,
    ChangePasswordRequest,
    UserLogin,
    UserRegister,
    UserResponse,
    UserUpdate,
    UserUpdateResponse,
    ApiKeyResponse,
    ApiKeyCreateResponse,
    EmailVerificationRequest,
    MessageResponse,
    RegistrationResponse,
)
from app.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    get_current_user,
    decode_token_with_version,
    bump_token_version,
)
from app.rate_limit import limiter, AUTH_RATE_LIMIT, AUTH_LOGIN_RATE_LIMIT

router = APIRouter(prefix="/auth", tags=["Authentication"])
settings = get_settings()
logger = logging.getLogger(__name__)

from app.audit import audit_log

REGISTRATION_MESSAGE = "Registration successful. Please check your email to verify your account before logging in."
VERIFICATION_RESEND_MESSAGE = "If the email is registered and unverified, a verification link has been sent."
VERIFICATION_REQUIRED_MESSAGE = "Please verify your email before logging in"


def _hash_verification_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _set_verification_token(user: User) -> str:
    token = secrets.token_urlsafe(32)
    user.verification_token_hash = _hash_verification_token(token)
    user.verification_token_created_at = datetime.now(timezone.utc)
    return token


def _verification_url(token: str) -> str:
    frontend_url = settings.FRONTEND_URL.rstrip("/")
    return f"{frontend_url}/verify-email?token={token}"


def _mail_configured() -> bool:
    return bool(settings.MAIL_SERVER and settings.MAIL_FROM)


def _dev_verification_url(token: str) -> str | None:
    if settings.ENVIRONMENT == "production" or _mail_configured():
        return None
    return f"/verify-email?token={token}"


async def _send_verification_email(user: User, token: str) -> None:
    if not _mail_configured():
        logger.warning("Email verification mail is not configured; skipping verification email for %s", user.email)
        return

    conf = ConnectionConfig(
        MAIL_USERNAME=settings.MAIL_USERNAME,
        MAIL_PASSWORD=settings.MAIL_PASSWORD,
        MAIL_FROM=settings.MAIL_FROM,
        MAIL_SERVER=settings.MAIL_SERVER,
        MAIL_PORT=settings.MAIL_PORT,
        MAIL_STARTTLS=settings.MAIL_STARTTLS,
        MAIL_SSL_TLS=settings.MAIL_SSL_TLS,
        USE_CREDENTIALS=bool(settings.MAIL_USERNAME and settings.MAIL_PASSWORD),
        VALIDATE_CERTS=True,
    )
    verification_link = _verification_url(token)
    message = MessageSchema(
        subject="Verify your Document AI Analyst account",
        recipients=[user.email],
        body=(
            f"<p>Hi {user.username},</p>"
            "<p>Please verify your email address before logging in to Document AI Analyst.</p>"
            f'<p><a href="{verification_link}">Verify your account</a></p>'
            "<p>This link expires in "
            f"{settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS} hours.</p>"
        ),
        subtype=MessageType.html,
    )

    try:
        await FastMail(conf).send_message(message)
    except Exception as exc:
        logger.error("Failed to send verification email for %s: %s", user.email, exc)

GOOGLE_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
GOOGLE_DRIVE_STATE_TOKEN_TYPE = "google_drive_oauth_state"


def _create_token_response(user: User) -> TokenResponse:
    tv = int(getattr(user, "token_version", 0) or 0)
    access_token = create_access_token(user.id, token_version=tv)
    refresh_token = create_refresh_token(user.id, token_version=tv)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.model_validate(user),
    )


def _verify_google_token(id_token_value: str) -> dict:
    if not settings.GOOGLE_CLIENT_ID:
        raise ExternalServiceException("Google", "Google sign-in is not configured")

    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token
    except ImportError as exc:
        raise ExternalServiceException("Google", "Google authentication dependency is not installed") from exc

    try:
        google_payload = id_token.verify_oauth2_token(
            id_token_value,
            Request(),
            settings.GOOGLE_CLIENT_ID,
        )
    except ValueError as exc:
        raise UnauthorizedException("Invalid Google credential") from exc

    email = google_payload.get("email")
    if not email or not google_payload.get("email_verified"):
        raise UnauthorizedException("Google account email is not verified")

    return google_payload


def _unique_google_username(email: str, db: Session) -> str:
    local_part = email.split("@", 1)[0]
    base = re.sub(r"[^a-zA-Z0-9_-]+", "-", local_part).strip("-_").lower() or "google-user"
    base = base[:70]
    candidate = base
    suffix = 1

    while db.query(User).filter(User.username == candidate).first():
        suffix += 1
        suffix_text = f"-{suffix}"
        candidate = f"{base[:80 - len(suffix_text)]}{suffix_text}"

    return candidate


@router.post("/register", response_model=RegistrationResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(AUTH_RATE_LIMIT)
async def register(request: Request, payload: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user account and send a verification email.

    Creates a new user in the database after validating that the username and
    email are not already taken. The password is hashed before storage. On
    success, a verification token is generated, stored as a hash, and emailed
    to the user before password login is allowed.

    Args:
        payload (UserRegister): The registration details including username, email, and password.
        db (Session, optional): Database session dependency. Defaults to Depends(get_db).

    Returns:
        RegistrationResponse: Message and email address for the pending verification.

    Raises:
        HTTPException: If the username is already taken (409 Conflict).
        HTTPException: If the email is already registered (409 Conflict).
    """
    # Check existing username
    if db.query(User).filter(User.username == payload.username).first():
        raise ConflictException("Username already taken")

    # Check existing email
    if db.query(User).filter(User.email == payload.email).first():
        raise ConflictException("Email already registered")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=UserRole.user,
        is_verified=not _mail_configured(),
    )
    token = _set_verification_token(user)
    db.add(user)
    db.commit()
    db.refresh(user)

    if _mail_configured():
        await _send_verification_email(user, token)

    audit_log(
        action="user.register",
        user_id=str(user.id),
        result="success",
        details={"email": user.email},
    )

    return RegistrationResponse(
        message=REGISTRATION_MESSAGE,
        email=user.email,
        verification_url=_dev_verification_url(token),
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit(AUTH_LOGIN_RATE_LIMIT)
def login(request: Request, payload: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate a user with email and password.

    Verifies that the provided email exists in the database and that the
    plain text password matches the stored hash. Upon successful authentication,
    generates new access and refresh tokens for the user session.

    Args:
        payload (UserLogin): User login data containing email and password.
        db (Session, optional): Database session dependency. Defaults to Depends(get_db).

    Returns:
        TokenResponse: An object containing:
            - access_token: JWT access token for API authentication.
            - refresh_token: JWT refresh token for obtaining new access tokens.
            - user: UserResponse object with the authenticated user's details.
    
    Raises:
        HTTPException: 401 Unauthorized if the email is not found or the
            password does not match the stored hash.
    """
    user = db.query(User).filter(User.email == payload.email).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        audit_log(
            action="user.login",
            result="failure",
            details={"email": payload.email, "reason": "invalid_credentials"},
        )
        raise UnauthorizedException("Invalid email or password")

    if not user.is_verified:
        audit_log(
            action="user.login",
            user_id=str(user.id),
            result="failure",
            details={"reason": "email_not_verified"},
        )
        raise AppException("FORBIDDEN", VERIFICATION_REQUIRED_MESSAGE, 403)

    user.last_login = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    audit_log(
        action="user.login",
        user_id=str(user.id),
        result="success",
        details={"method": "password"},
    )

    return _create_token_response(user)


@router.post("/dev-login", response_model=TokenResponse)
@limiter.limit(AUTH_LOGIN_RATE_LIMIT)
def dev_login(request: Request, db: Session = Depends(get_db)):
    """Auto sign-in for local development only.

    Creates a verified ``localdev`` user if needed and returns JWTs so the
    frontend can skip the login form. Disabled when ``ENVIRONMENT=production``
    or ``DEV_AUTH_BYPASS=false``.
    """
    if settings.ENVIRONMENT == "production" or not settings.DEV_AUTH_BYPASS:
        raise AppException(
            "FORBIDDEN",
            "Dev auth bypass is disabled outside local development.",
            403,
        )

    email = "dev@genova.ai"
    username = "localdev"
    user = (
        db.query(User).filter(User.email == email).first()
        or db.query(User).filter(User.email == "localdev@localhost").first()
    )
    if user and user.email != email:
        user.email = email
        db.commit()
        db.refresh(user)
    if not user:
        user = User(
            username=username,
            email=email,
            hashed_password=hash_password(secrets.token_urlsafe(32)),
            role=UserRole.user,
            is_verified=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    elif not user.is_verified:
        user.is_verified = True
        db.commit()
        db.refresh(user)

    user.last_login = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    audit_log(
        action="user.login",
        user_id=str(user.id),
        result="success",
        details={"method": "dev_bypass"},
    )
    return _create_token_response(user)


@router.post("/google", response_model=TokenResponse)
@limiter.limit(AUTH_LOGIN_RATE_LIMIT)
def login_with_google(request: Request, payload: GoogleLoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate with a Google Identity Services ID token.

    Existing users are matched by verified email. New Google users get an
    internal username derived from the email prefix and an unusable random
    password hash so password login remains opt-in through the normal flow.
    """
    google_payload = _verify_google_token(payload.id_token)
    email = str(google_payload["email"]).lower()

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            username=_unique_google_username(email, db),
            email=email,
            hashed_password=hash_password(secrets.token_urlsafe(32)),
            is_verified=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    elif not user.is_verified:
        user.is_verified = True
        user.verification_token_hash = None
        user.verification_token_created_at = None

    user.last_login = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    return _create_token_response(user)


@router.get("/verify", response_model=MessageResponse)
def verify_email(token: str, db: Session = Depends(get_db)):
    """Verify a registered user's email address using an emailed token."""
    token_hash = _hash_verification_token(token)
    user = db.query(User).filter(User.verification_token_hash == token_hash).first()
    if not user:
        raise ValidationException("Invalid or expired verification token")

    created_at = user.verification_token_created_at
    if created_at and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    expires_at = (created_at or datetime.now(timezone.utc)) + timedelta(
        hours=settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS
    )
    if expires_at < datetime.now(timezone.utc):
        user.verification_token_hash = None
        user.verification_token_created_at = None
        db.commit()
        raise ValidationException("Invalid or expired verification token")

    user.is_verified = True
    user.verification_token_hash = None
    user.verification_token_created_at = None
    db.commit()

    return MessageResponse(message="Email verified successfully. You can now log in.")


@router.post("/resend-verification", response_model=MessageResponse)
@limiter.limit(AUTH_RATE_LIMIT)
async def resend_verification(request: Request, payload: EmailVerificationRequest, db: Session = Depends(get_db)):
    """Resend an email verification link for an unverified user."""
    user = db.query(User).filter(User.email == payload.email).first()
    if user and not user.is_verified:
        token = _set_verification_token(user)
        db.commit()
        db.refresh(user)
        await _send_verification_email(user, token)

    return MessageResponse(message=VERIFICATION_RESEND_MESSAGE)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit(AUTH_RATE_LIMIT)
def refresh_token(request: Request, payload: RefreshRequest, db: Session = Depends(get_db)):
    """
    Refresh both access and refresh tokens using a valid refresh token.

    Decodes the provided refresh token to extract the user ID. If the token
    is valid, the token_version matches, and the user still exists, a new
    pair of access and refresh tokens is generated and returned.
    """
    decoded = decode_token_with_version(payload.refresh_token, token_type="refresh")
    if not decoded:
        raise UnauthorizedException("Invalid or expired refresh token")

    user_id, token_tv = decoded
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise UnauthorizedException("User not found")

    current_tv = int(getattr(user, "token_version", 0) or 0)
    if token_tv != current_tv:
        raise UnauthorizedException("Invalid or expired refresh token")

    return _create_token_response(user)


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    """Retrieve the profile of the currently authenticated user.

    Returns the user object associated with the valid access token provided
    in the request. This endpoint is useful for fetching the current user's
    information after login or token refresh.

    Args:
        user: The currently authenticated user, obtained from the `get_current_user` dependency.

    Returns:
        UserResponse: The authenticated user's public profile data, including
        id, username, email, and any other exposed fields.

    Note:
        This endpoint relies on the `get_current_user` dependency, which
        will automatically return a 401 Unauthorized response if the access
        token is missing, invalid, or expired. Therefore, this function
        itself does not need to raise any HTTP exceptions.
    """
    return UserResponse.model_validate(user)


@router.put("/update")
def update_user_info(payload:UserUpdate,
                    user: User = Depends(get_current_user),
                    db: Session = Depends(get_db))-> UserUpdateResponse:
    """Update the current user's profile information.

    Allows an authenticated user to change their username and/or email address.
    At least one of `username` or `email` must be provided. If a new value
    is supplied, it is checked for uniqueness against existing users. On
    success, the updated user record is returned.

    Args:
        payload: UserUpdate object containing optional `username` and `email` fields to update.
        user: The currently authenticated user, obtained from the `get_current_user` dependency.
        db: SQLAlchemy database session, obtained from the dependency.

    Returns:
        UserUpdateResponse: The updated user profile data (same structure as
        the database model, exposed through the response model).

    Raises:
        HTTPException: 400 if:
            - Neither `username` nor `email` is provided.
            - The new username is already taken.
            - The new email is already registered (checks both fields).
            - A database error occurs (e.g., integrity or connection issue).

    Note:
        The function commits changes to the database and refreshes the user
        instance before returning. Any `SQLAlchemyError` triggers a rollback
        and a 400 response.
    """
    if payload.username is None and payload.email is None:
        raise ValidationException("At least one of username or email must be provided")

    try:
        if payload.username:
            existing_user = db.execute(select(User).where(User.username == payload.username)).scalar_one_or_none()

            if existing_user:
                raise ValidationException("Username already exists")
            user.username = payload.username
        if payload.email:
            existing_user = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()

            if existing_user:
                raise ValidationException("Email already exists")
            user.email = payload.email
        db.commit()
        db.refresh(user)
        return user
    except SQLAlchemyError:
        db.rollback()
        raise ValidationException("Database error")

@router.put("/password")
def update_password(payload:UpdatePassword,
                    user: User = Depends(get_current_user),
                    db: Session = Depends(get_db))-> UpdatePasswordResponse:
    """Update the authenticated user's password.

    Validates that both the new password and confirmation are provided and
    match each other. If valid, the password is hashed and stored. The user
    record is then committed to the database.

    Args:
        payload: UpdatePassword object containing `password` and `confirm_password` fields.
        user: The currently authenticated user, obtained from the `get_current_user` dependency.
        db: SQLAlchemy database session, obtained from the dependency.

    Returns:
        UpdatePasswordResponse: The updated user object (typically containing 
        user details excluding the password hash).

    Raises:
        HTTPException: 400 if:
            - Either `password` or `confirm_password` is missing or empty.
            - The two password fields do not match.
            - A database error (SQLAlchemyError) occurs during commit.

    Note:
        The function hashes the password using `hash_password()` before
        saving. Any `SQLAlchemyError` triggers a rollback and raises a 400
        response.
    """
    if not payload.password and not payload.confirm_password:
        raise ValidationException("Password and confirm_password are required")
    if len(payload.password) == 0 and len(payload.confirm_password) == 0:
        raise ValidationException("Password and confirm_password are required")
    if payload.password != payload.confirm_password:
        raise ValidationException("Password and confirm_password are different")
    # Harden legacy endpoint: require proving the current password via the
    # dedicated change-password flow. Accept payload.current_password if present
    # on UpdatePassword; otherwise reject to prevent session-only takeover.
    current_password = getattr(payload, "current_password", None)
    if not current_password or not verify_password(current_password, user.hashed_password):
        raise UnauthorizedException(
            "Current password is required. Use POST /auth/change-password."
        )

    try:
        hashed_password = hash_password(payload.password)
        user.hashed_password = hashed_password
        bump_token_version(user, db)
        return user
    except SQLAlchemyError:
        db.rollback()
        raise ValidationException("Database error")


@router.post("/change-password", response_model=MessageResponse)
def change_password(
    payload: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Securely rotate the authenticated user's password.

    Validates that the provided current password matches the stored hash
    before updating to the new password. This is the secure, recommended
    way for users to change their own password since it confirms the
    requester actually knows the existing credentials. Also invalidates
    all outstanding JWTs by bumping token_version.
    """
    if not verify_password(payload.current_password, user.hashed_password):
        raise UnauthorizedException("Current password is incorrect")

    try:
        user.hashed_password = hash_password(payload.new_password)
        bump_token_version(user, db)
    except SQLAlchemyError:
        db.rollback()
        raise ValidationException("Database error")

    audit_log(
        action="user.change_password",
        user_id=str(user.id),
        result="success",
    )

    return MessageResponse(message="Password updated successfully")


@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    body: dict = Body(None),
):
    """Create a new API key for the authenticated user."""
    name = (body or {}).get("name", "default")
    raw_key = "pdf_rag_" + secrets.token_hex(24)
    hashed_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    api_key = ApiKey(
        user_id=user.id,
        name=name,
        key_prefix=raw_key[:15],
        hashed_key=hashed_key,
        is_active=True,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    audit_log(
        action="api_key.create",
        user_id=str(user.id),
        result="success",
        resource=str(api_key.id),
        details={"name": name},
    )

    return ApiKeyCreateResponse(
        id=str(api_key.id),
        name=api_key.name,
        key_preview=api_key.key_prefix,
        created_at=api_key.created_at,
        raw_key=raw_key,
    )

@router.get("/api-keys", response_model=List[ApiKeyResponse])
def list_api_keys(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all API keys for the authenticated user."""
    keys = db.query(ApiKey).filter(ApiKey.user_id == user.id, ApiKey.is_active == True).all()
    return [
        ApiKeyResponse(
            id=str(k.id),
            name=k.name,
            key_preview=k.key_prefix,
            created_at=k.created_at,
        )
        for k in keys
    ]

@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_api_key(key_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Revoke an API key."""
    api_key = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.user_id == user.id).first()
    if not api_key:
        raise NotFoundException("API key")

    db.delete(api_key)
    db.commit()

    audit_log(
        action="api_key.delete",
        user_id=str(user.id),
        result="success",
        resource=key_id,
    )

    return None

@router.get("/config")
def get_auth_config():
    """Return public configuration for auth providers"""
    return {
        "google_client_id": settings.GOOGLE_CLIENT_ID
    }


def _unique_google_username(email: str, db: Session) -> str:
    """
    Generate a unique username based on the email.
    """
    base = email.split("@")[0]
    base = re.sub(r"[^a-zA-Z0-9_-]", "", base)
    base = base[:70]
    candidate = base
    suffix = 1

    while db.query(User).filter(User.username == candidate).first():
        suffix += 1
        suffix_text = f"-{suffix}"
        candidate = f"{base[:80 - len(suffix_text)]}{suffix_text}"

    return candidate


def _require_google_drive_config() -> None:
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        raise ExternalServiceException("Google Drive", "Google Drive OAuth is not configured")


def _create_google_drive_state(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": GOOGLE_DRIVE_STATE_TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(minutes=10),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def _decode_google_drive_state(state_token: str) -> str | None:
    try:
        payload = jwt.decode(
            state_token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.InvalidTokenError:
        return None

    if payload.get("type") != GOOGLE_DRIVE_STATE_TOKEN_TYPE:
        return None
    return payload.get("sub")


def _google_drive_flow(state: str | None = None) -> Flow:
    _require_google_drive_config()
    return Flow.from_client_config(
        {
            "web": {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.GOOGLE_DRIVE_REDIRECT_URI],
            }
        },
        scopes=GOOGLE_DRIVE_SCOPES,
        state=state,
        redirect_uri=settings.GOOGLE_DRIVE_REDIRECT_URI,
    )


def _google_drive_callback_page(title: str, message: str, success: bool) -> HTMLResponse:
    color = "#16a34a" if success else "#dc2626"
    safe_title = escape(title)
    safe_message = escape(message)
    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
          <head>
            <title>{safe_title}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1" />
          </head>
          <body style="font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5;">
            <h1 style="color: {color};">{safe_title}</h1>
            <p>{safe_message}</p>
            <p>You can close this tab and return to Document AI Analyst.</p>
          </body>
        </html>
        """,
        status_code=status.HTTP_200_OK if success else status.HTTP_400_BAD_REQUEST,
    )


@router.get("/google-drive/connect", response_model=GoogleDriveAuthUrlResponse)
def connect_google_drive(user: User = Depends(get_current_user)):
    """
    Create a Google OAuth consent URL for read-only Drive access.

    The OAuth state is a short-lived signed token containing the current user
    id, so the callback can store the refresh token without trusting client
    input or requiring a browser Authorization header.
    """
    state_token = _create_google_drive_state(user.id)
    flow = _google_drive_flow(state_token)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return GoogleDriveAuthUrlResponse(auth_url=auth_url)


@router.get("/google-drive/callback")
def google_drive_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Handle Google's OAuth callback and persist the encrypted refresh token."""
    if error:
        return _google_drive_callback_page("Google Drive not connected", error, False)

    if not code or not state:
        return _google_drive_callback_page(
            "Google Drive not connected",
            "Missing Google OAuth callback data.",
            False,
        )

    user_id = _decode_google_drive_state(state)
    if not user_id:
        return _google_drive_callback_page(
            "Google Drive not connected",
            "Invalid or expired Google OAuth state.",
            False,
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return _google_drive_callback_page("Google Drive not connected", "User not found.", False)

    try:
        flow = _google_drive_flow(state)
        flow.fetch_token(code=code)
    except Exception:
        return _google_drive_callback_page(
            "Google Drive not connected",
            "Failed to connect Google Drive.",
            False,
        )

    refresh_token = flow.credentials.refresh_token
    if not refresh_token:
        return _google_drive_callback_page(
            "Google Drive not connected",
            "Google did not return a refresh token. Revoke access and try again.",
            False,
        )

    user.google_refresh_token = refresh_token
    db.commit()
    db.refresh(user)

    return _google_drive_callback_page(
        "Google Drive connected",
        "Read-only Google Drive access was granted successfully.",
        True,
    )


@router.get("/google-drive/status", response_model=GoogleDriveStatusResponse)
def google_drive_status(user: User = Depends(get_current_user)):
    """Return whether the current user has connected Google Drive."""
    return GoogleDriveStatusResponse(connected=bool(user.google_refresh_token))


@router.delete("/google-drive/disconnect", response_model=GoogleDriveStatusResponse)
def disconnect_google_drive(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove the stored Google Drive refresh token for the current user."""
    user.google_refresh_token = None
    db.commit()
    return GoogleDriveStatusResponse(connected=False)


@router.post("/logout")
def logout(
    response: Response,
    request: Request,
    payload: Optional[RefreshRequest] = Body(default=None),
    db: Session = Depends(get_db),
):
    """
    Invalidate the caller's session.

    Prefers Authorization bearer access token; falls back to refresh_token in
    the body. Bumps ``token_version`` so outstanding JWTs stop working, then
    clears legacy cookies if present.
    """
    user: Optional[User] = None

    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        raw = auth_header.split(" ", 1)[1].strip()
        if not raw.startswith("pdf_rag_"):
            decoded = decode_token_with_version(raw, token_type="access")
            if decoded:
                user_id, token_tv = decoded
                candidate = db.query(User).filter(User.id == user_id).first()
                if candidate and int(getattr(candidate, "token_version", 0) or 0) == token_tv:
                    user = candidate

    if user is None and payload and payload.refresh_token:
        decoded = decode_token_with_version(payload.refresh_token, token_type="refresh")
        if decoded:
            user_id, token_tv = decoded
            candidate = db.query(User).filter(User.id == user_id).first()
            if candidate and int(getattr(candidate, "token_version", 0) or 0) == token_tv:
                user = candidate

    if user is not None:
        bump_token_version(user, db)
        audit_log(
            action="user.logout",
            user_id=str(user.id),
            result="success",
        )

    response.delete_cookie(key="access_token")
    response.delete_cookie(key="refresh_token")
    return {"message": "Successfully logged out"}