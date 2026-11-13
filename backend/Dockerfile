FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY static ./static

RUN adduser --disabled-password --gecos '' appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8787/api/health || exit 1

EXPOSE 8787
USER appuser
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8787"]
