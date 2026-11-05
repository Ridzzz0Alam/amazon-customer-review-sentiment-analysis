# syntax=docker/dockerfile:1.7
# Multi-stage build following the astral-sh/uv Docker guide:
# dependencies are resolved with uv in the builder, the runtime image has no uv and runs as non-root.

FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 UV_PYTHON_DOWNLOADS=0
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
COPY pyproject.toml uv.lock README.md params.yaml ./
COPY src ./src
COPY models ./models
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

FROM python:3.13-slim-trixie
RUN groupadd --system --gid 999 app && useradd --system --gid 999 --uid 999 --create-home app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    SENTIMENT_PREDICTION_LOG_PATH=/app/logs/predictions.jsonl
WORKDIR /app
RUN mkdir -p /app/logs && chown app:app /app/logs
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["fastapi", "run", "src/sentiment_mlops/api/main.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
