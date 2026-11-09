"""
Deep health check endpoint — verifies DB, Redis, Celery, and ChromaDB.

GET /api/v1/health/status
- No authentication required (monitoring tools must reach it freely).
- Never raises a 5xx — always returns 200 with per-component status so
  load balancers and uptime monitors can parse the JSON body.
- Never returns connection strings, hostnames from exceptions, worker
  inventory, or internal service URLs to anonymous callers.
"""
import logging
import time
from typing import Dict, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


def _safe_failure(component: str, exc: Exception) -> Dict[str, Any]:
    """Log the real error; return a non-leaky public payload."""
    logger.warning("%s health check failed: %s", component, exc)
    return {"status": "down"}


def _check_database() -> Dict[str, Any]:
    """Ping the configured database (MySQL or SQLite)."""
    start = time.perf_counter()
    try:
        from app.database import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"status": "up", "latency_ms": latency_ms}
    except Exception as exc:
        return _safe_failure("DB", exc)


def _check_redis() -> Dict[str, Any]:
    """Ping the Redis broker used by Celery and the response cache."""
    start = time.perf_counter()
    try:
        from app.cache import _get_redis, _redis_last_error
        from app.config import get_settings

        configured = bool((get_settings().REDIS_URL or "").strip())
        r = _get_redis()
        if r is None:
            # Do not echo _redis_last_error — it can include host/auth fragments.
            if _redis_last_error:
                logger.warning("Redis unavailable: %s", _redis_last_error)
            return {
                "status": "unavailable",
                "detail": (
                    "REDIS_URL not configured"
                    if not configured
                    else "Redis configured but unreachable"
                ),
            }
        r.ping()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"status": "up", "latency_ms": latency_ms}
    except Exception as exc:
        return _safe_failure("Redis", exc)


def _check_celery() -> Dict[str, Any]:
    """
    Verify at least one Celery worker is reachable via the inspect API.
    Uses a short timeout so the endpoint stays fast even when workers
    are offline.
    """
    start = time.perf_counter()
    try:
        from app.celery_app import celery_app

        inspector = celery_app.control.inspect(timeout=2.0)
        active = inspector.active()         # None if no workers respond
        ping = inspector.ping()

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if not ping:
            return {
                "status": "down",
                "detail": "No Celery workers responded to ping",
                "latency_ms": latency_ms,
            }

        worker_count = len(ping.keys())
        active_task_count = sum(
            len(tasks) for tasks in (active or {}).values()
        )
        # Omit worker hostnames from the public payload.
        return {
            "status": "up",
            "workers": worker_count,
            "active_tasks": active_task_count,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        return _safe_failure("Celery", exc)


def _check_chromadb() -> Dict[str, Any]:
    """Ping the ChromaDB vector store."""
    start = time.perf_counter()
    try:
        from app.rag.vectordb.vectorstore import get_chroma_client
        client = get_chroma_client()
        client.heartbeat()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"status": "up", "latency_ms": latency_ms}
    except Exception as exc:
        return _safe_failure("ChromaDB", exc)


def _check_ollama() -> Dict[str, Any]:
    """Ping local Ollama when configured as LLM or embedding provider."""
    from app.config import get_settings

    settings = get_settings()
    if not (settings.uses_ollama_llm or settings.uses_ollama_embeddings):
        return {"status": "skipped", "detail": "Ollama not configured as a provider"}

    start = time.perf_counter()
    try:
        import httpx

        url = settings.OLLAMA_BASE_URL.rstrip("/")
        with httpx.Client(timeout=5.0) as client:
            version = client.get(f"{url}/api/version")
            version.raise_for_status()
            tags = client.get(f"{url}/api/tags")
            tags.raise_for_status()
            names = sorted(
                {m.get("name", "") for m in tags.json().get("models", []) if m.get("name")}
            )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        needed = []
        if settings.uses_ollama_llm:
            needed.append(settings.LLM_MODEL)
        if settings.uses_ollama_embeddings:
            needed.append(settings.EMBEDDING_MODEL)

        def _installed(model: str) -> bool:
            base = model.split(":")[0]
            return model in names or any(n == model or n.startswith(base + ":") for n in names)

        missing = [m for m in needed if not _installed(m)]
        status = "up" if not missing else "degraded"
        payload: Dict[str, Any] = {
            "status": status,
            "latency_ms": latency_ms,
            "required_models": needed,
            "missing_models": missing,
        }
        # Internal base URL + full model inventory only outside production.
        if settings.ENVIRONMENT != "production":
            payload["base_url"] = url
            payload["installed_models"] = names
        return payload
    except Exception as exc:
        return _safe_failure("Ollama", exc)


@router.get(
    "/status",
    summary="Deep system health check",
    description=(
        "Returns connection status for MySQL/SQLite, Redis, "
        "Celery workers, ChromaDB, and Ollama (when configured). "
        "Always returns HTTP 200 — inspect the `overall` field to "
        "determine system health."
    ),
)
def health_status():
    """
    Deep health check — verifies all backend dependencies.

    Returns HTTP 200 always. Callers should inspect the ``overall`` field:
    - ``healthy``  — all required components up.
    - ``degraded`` — one or more components down or unavailable.

    Redis/Celery may be unavailable locally (eager fallback); that alone
    does not block RAG when Ollama + the relational DB + ChromaDB are healthy.
    """
    checks = {
        "database": _check_database(),
        "redis":    _check_redis(),
        "celery":   _check_celery(),
        "chromadb": _check_chromadb(),
        "ollama":   _check_ollama(),
    }

    # Core local path: DB + Chroma + Ollama (when enabled). Redis/Celery optional.
    core_keys = ["database", "chromadb"]
    if checks["ollama"].get("status") != "skipped":
        core_keys.append("ollama")

    core_ok = all(
        checks[k].get("status") in {"up", "degraded"} and checks[k].get("status") != "down"
        for k in core_keys
    )
    # Treat ollama "degraded" (missing model) as not fully healthy
    ollama_status = checks["ollama"].get("status")
    if ollama_status == "degraded":
        core_ok = False

    overall = "healthy" if core_ok and ollama_status in {"up", "skipped"} else "degraded"

    return JSONResponse(
        status_code=200,
        content={
            "overall": overall,
            "components": checks,
        },
    )
