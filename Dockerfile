FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install all dependencies (API + UI)
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy source
COPY --chown=appuser:appuser . .

# Create runtime dirs
RUN mkdir -p /app/models/saved /app/data/splits /app/data/processed \
    && chown -R appuser:appuser /app/models /app/data

USER appuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# UI will call the API on localhost
ENV API_BASE_URL=http://localhost:8000

EXPOSE 7860

# Start FastAPI on :8000 in background, wait for it, then start Gradio on :7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info & sleep 8 && python app.py"]