FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --shell /bin/bash appuser

COPY --chown=appuser:appuser . .

RUN mkdir -p /app/models/saved /app/data/splits /app/data/processed \
    && chown -R appuser:appuser /app/models /app/data

USER appuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV API_BASE_URL=http://localhost:8000

EXPOSE 7860

# Start FastAPI in background on port 8000, then launch Gradio on 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &\n sleep 5 &&\n python app.py"]