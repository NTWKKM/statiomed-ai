# ==============================================================================
# StatioMed AI - Dockerfile for HuggingFace Spaces Deployment
# ==============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /build/deps && \
    pip install --target=/build/deps --no-cache-dir --upgrade pip "setuptools>=80.10.1" "wheel>=0.46.3"

COPY requirements-prod.txt ./
RUN pip install --target=/build/deps --no-cache-dir -r requirements-prod.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="StatioMed AI" \
      org.opencontainers.image.description="Agentic Medical Statistical Analysis & Study Design Platform" \
      org.opencontainers.image.source="https://github.com/NTWKKM/statiomed-ai" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/deps \
    HOME=/home/appuser

WORKDIR /app

COPY --from=builder /build/deps /app/deps

RUN (id -u appuser >/dev/null 2>&1 || useradd -m -u 1000 appuser) && \
    chown -R appuser:appuser /app

RUN apt-get update && \
    apt-get upgrade -y && \
    pip install --no-cache-dir --upgrade "pip>=25.3" && \
    PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright python -m playwright install --with-deps chromium && \
    chown -R appuser:appuser /home/appuser/.cache && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860', timeout=3)" || exit 1

CMD ["python", "-m", "gunicorn", \
  "-k", "uvicorn.workers.UvicornWorker", \
  "-w", "2", \
  "--timeout", "120", \
  "--graceful-timeout", "30", \
  "--worker-tmp-dir", "/dev/shm", \
  "--bind", "0.0.0.0:7860", \
  "--preload", \
  "asgi:app"]
