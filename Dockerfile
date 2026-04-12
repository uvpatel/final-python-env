FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    ENABLE_GRADIO_DEMO=false \
    ENABLE_WEB_INTERFACE=false

WORKDIR /app

COPY server/requirements.runtime.txt /tmp/requirements.runtime.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    python -m pip install --upgrade pip setuptools && \
    pip install -r /tmp/requirements.runtime.txt

COPY --chown=appuser:appuser . /app

RUN pip install --no-deps .

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
