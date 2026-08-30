# ==============================================================================
# StatioMed AI - Production Dockerfile for Hugging Face Spaces & Native Gradio
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
    apt-get install -y --no-install-recommends gcc g++ libgomp1 && \
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
      org.opencontainers.image.description="Agentic Clinical Research & Biostatistical Co-Pilot (Native Gradio 6.x)" \
      org.opencontainers.image.source="https://github.com/NTWKKM/statiomed-ai" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/deps \
    HOME=/home/appuser \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860" \
    PORT="7860" \
    SYSTEM="spaces"

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /build/deps /app/deps

# Create standard non-root user (UID 1000 required for Hugging Face Spaces)
RUN (id -u appuser >/dev/null 2>&1 || useradd -m -u 1000 appuser) && \
    mkdir -p /app /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser

# Copy application source code
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860', timeout=3)" || exit 1

# Launch Pure Native Gradio App
CMD ["python", "app.py"]
