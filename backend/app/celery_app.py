import os
import logging
from celery import Celery
from celery.signals import worker_process_init

logger = logging.getLogger(__name__)

# Always mute pdfminer/HTTP DEBUG on import (API + worker). Full Loguru setup
# for Celery workers is deferred to worker_process_init — workers never import
# app.main, so setup_logging() would otherwise never run there.
try:
    from app.observability import _quiet_noisy_loggers

    _quiet_noisy_loggers(logging)
except Exception:
    for name in ("pdfminer", "PIL", "fontTools", "botocore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


@worker_process_init.connect
def _configure_worker_logging(**_kwargs):
    try:
        from app.observability import setup_logging

        setup_logging()
    except Exception as exc:
        logger.warning("Celery worker logging setup failed: %s", exc)


broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
backend_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Check if Redis is reachable to determine if we should fall back to eager mode
redis_available = False
if broker_url.startswith("redis://"):
    try:
        import redis
        r = redis.Redis.from_url(broker_url, socket_timeout=1)
        r.ping()
        redis_available = True
    except Exception:
        redis_available = False

always_eager = not redis_available or os.getenv("CELERY_TASK_ALWAYS_EAGER", "False").lower() in ("true", "1")

if always_eager:
    logger.warning(
        "Redis broker is not reachable (or CELERY_TASK_ALWAYS_EAGER=1). "
        "Ingest runs in-process via FastAPI BackgroundTasks — not a separate Celery worker."
    )

# Initialize the Celery application instance
celery_app = Celery(
    "worker",
    broker=broker_url,
    backend=backend_url
)

# Optional configuration updates for reliable serialization
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_always_eager=always_eager,
)

# Tell Celery to discover background tasks dynamically to break circular loops
celery_app.autodiscover_tasks(["app"])