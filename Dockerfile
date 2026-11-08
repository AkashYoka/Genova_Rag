# syntax=docker/dockerfile:1

# --------------------------------------------------------
# Stage 1: Build Python dependencies in an isolated venv
# --------------------------------------------------------
FROM python:3.11-slim AS python-builder

# uv is builder-only (not copied into the runtime image)
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Copy into venv instead of hardlinking (safer across Docker layers)
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    libxml2-dev \
    libxslt1-dev \
    pkg-config \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Same path convention as before (/opt/venv)
RUN uv venv "$VIRTUAL_ENV"

# requirements.txt first for layer cache when app code changes
COPY backend/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + && \
    find /opt/venv -type f -name "*.pyc" -delete

# --------------------------------------------------------
# Stage 2: Runtime image with only backend code and dependencies
# --------------------------------------------------------
FROM python:3.11-slim

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Non-root runtime user (uid 1000 — match EFS access point on ECS)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Runtime-only system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY --from=python-builder /opt/venv /opt/venv

# Copy backend code
COPY backend/app ./app
COPY backend/__init__.py ./backend/__init__.py

# Create data directories with proper permissions
RUN mkdir -p /app/data/uploads /app/data/chroma_db /app/data/graphs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
