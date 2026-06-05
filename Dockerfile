FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt ./
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements-api.txt

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Movie Recommendation API"
LABEL org.opencontainers.image.description="SVD recommender REST API"
LABEL org.opencontainers.image.source="https://github.com/Padmanav-Mohanty/Movie-Recommendation-System"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY --chown=appuser:appuser . .
RUN mkdir -p /app/models/saved \
    && mkdir -p /app/data/splits \
    && mkdir -p /app/data/processed \
    && chown -R appuser:appuser /app/models \
    && chown -R appuser:appuser /app/data

USER appuser

ENV PORT=8000
ENV ENV=production
ENV WEB_CONCURRENCY=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1} --log-level info"]