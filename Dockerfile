dockerfile# ── Clinical Data Pipeline — Production Dockerfile ────────────────────────────
#
# Multi-stage build for the SDTM/ADaM/TLF clinical data pipeline.
# Includes all ML dependencies: TensorFlow, scikit-learn, lifelines
#
# Author : Girish Rajeev
#          Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Production runtime ───────────────────────────────────────────────
FROM python:3.10-slim AS production

LABEL maintainer="Girish Rajeev"
LABEL description="Clinical Data Pipeline — SDTM/ADaM/TLF with LLM and ML"
LABEL version="1.0.0"

WORKDIR /app

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY tests/ ./tests/

RUN mkdir -p /app/data /app/outputs /app/logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import src.sdtm.base; print('healthy')" || exit 1

CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
