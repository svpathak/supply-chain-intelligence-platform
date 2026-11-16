# Small, reproducible runtime for the API and one-shot pipeline.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/raw ./data/raw
COPY artifacts ./artifacts
COPY reports ./reports
COPY dq_check.py ./dq_check.py
COPY README.md ./README.md

RUN mkdir -p /app/var /app/logs && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
