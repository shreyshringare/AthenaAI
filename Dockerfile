# Multi-stage build: keep the final image slim by separating build from runtime.
# Stage 1: install uv and resolve dependencies
# Stage 2: copy installed packages + source, run as non-root

# ---- build stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency manifest first (layer cache hit when source changes but deps don't)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install into a venv at /app/.venv
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -e ".[dev]"

# ---- runtime stage ----
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash athena
WORKDIR /app

# Copy installed venv from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /build/src /app/src
COPY --from=builder /build/pyproject.toml /app/pyproject.toml

# Activate venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_FORMAT=json

USER athena

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "athenai.gateway.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
