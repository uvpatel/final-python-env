FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml __init__.py client.py compat.py models.py inference.py /app/
COPY server /app/server
COPY tasks /app/tasks
COPY graders /app/graders

RUN python -m pip install --upgrade pip && \
    pip install .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
