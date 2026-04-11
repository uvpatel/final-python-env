FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md DEMO_SCRIPT.md openenv.yaml __init__.py client.py compat.py openenv_models.py inference.py triage.py triage_catalog.py triage_models.py launch.py /app/
COPY api /app/api
COPY app /app/app
COPY analyzers /app/analyzers
COPY models /app/models
COPY schemas /app/schemas
COPY server /app/server
COPY services /app/services
COPY tasks /app/tasks
COPY utils /app/utils
COPY graders /app/graders

RUN python -m pip install --upgrade pip && \
    pip install .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000', timeout=3).read()"

CMD ["python", "launch.py"]
