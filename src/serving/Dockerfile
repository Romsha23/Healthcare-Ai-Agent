Now let's add the Dockerfile:

Stay inside the serving folder
Click "Add file" → "Create new file"
Type in the filename box:

Dockerfile

Paste this code:

dockerfile# ── Healthcare AI Platform — Production Dockerfile ────────────────────────────
#
# Multi-stage build for minimal production image size.
# Stage 1: Build dependencies
# Stage 2: Production runtime
#
# Author : Girish Rajeev
#          Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Production runtime ───────────────────────────────────────────────
FROM python:3.10-slim AS production

LABEL maintainer="Girish Rajeev <girish.rajeev@example.com>"
LABEL description="Healthcare AI Platform — RAG, Agents, Recommender, Serving"
LABEL version="1.0.0"

WORKDIR /app

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY src/ ./src/
COPY tests/ ./tests/

# Create directories for persistent data
RUN mkdir -p /app/chroma_db /app/data /app/logs /app/embedding_cache

# Environment defaults (override via docker-compose or K8s secrets)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV ENV=production
ENV CHROMA_DB_DIR=/app/chroma_db
ENV LOG_LEVEL=info
ENV RATE_LIMIT_REQUESTS=60
ENV RATE_LIMIT_WINDOW_SEC=60

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI with uvicorn
CMD ["python", "-m", "uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]

Click "Commit changes" → "Commit changes" ✅

Then immediately add docker-compose.yml in the same folder:

Click "Add file" → "Create new file"
Type in the filename box:

docker-compose.yml

Paste this code:

yaml# Healthcare AI Platform — Docker Compose
# Full local development and testing stack
#
# Services:
#   api       — FastAPI healthcare AI platform
#   chromadb  — Vector store for RAG pipeline
#   redis     — Rate limiting and caching (optional)
#
# Usage:
#   docker-compose up -d
#   docker-compose logs -f api
#   docker-compose down
#
# Author: Girish Rajeev

version: "3.9"

services:

  # ── Healthcare AI API ──────────────────────────────────────────────────────
  api:
    build:
      context: ../..
      dockerfile: src/serving/Dockerfile
      target: production
    container_name: healthcare-ai-api
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - API_KEYS=${API_KEYS:-dev-key-123,test-key-456}
      - CHROMA_DB_DIR=/app/chroma_db
      - ENV=${ENV:-development}
      - PORT=8000
      - LOG_LEVEL=${LOG_LEVEL:-info}
      - RATE_LIMIT_REQUESTS=${RATE_LIMIT_REQUESTS:-60}
      - RATE_LIMIT_WINDOW_SEC=${RATE_LIMIT_WINDOW_SEC:-60}
    volumes:
      - chroma_data:/app/chroma_db
      - app_logs:/app/logs
      - embedding_cache:/app/embedding_cache
    depends_on:
      chromadb:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    networks:
      - healthcare-ai-net

  # ── ChromaDB Vector Store ──────────────────────────────────────────────────
  chromadb:
    image: chromadb/chroma:latest
    container_name: healthcare-ai-chromadb
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
      - ANONYMIZED_TELEMETRY=false
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 20s
      timeout: 5s
      retries: 3
      start_period: 20s
    restart: unless-stopped
    networks:
      - healthcare-ai-net

  # ── Development only: API with hot reload ─────────────────────────────────
  api-dev:
    build:
      context: ../..
      dockerfile: src/serving/Dockerfile
      target: builder
    container_name: healthcare-ai-dev
    ports:
      - "8080:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - API_KEYS=dev-key-123
      - CHROMA_DB_DIR=/app/chroma_db
      - ENV=development
      - LOG_LEVEL=debug
    volumes:
      - ../../src:/app/src:ro
      - chroma_data:/app/chroma_db
      - embedding_cache:/app/embedding_cache
    command: >
      python -m uvicorn src.serving.api:app
      --host 0.0.0.0
      --port 8000
      --reload
      --log-level debug
    depends_on:
      - chromadb
    profiles:
      - dev
    networks:
      - healthcare-ai-net

# ── Volumes ────────────────────────────────────────────────────────────────────
volumes:
  chroma_data:
    driver: local
  app_logs:
    driver: local
  embedding_cache:
    driver: local

# ── Networks ───────────────────────────────────────────────────────────────────
networks:
  healthcare-ai-net:
    driver: bridge
