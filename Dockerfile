# ---------------------------------------------------------------------------
# Stage 1 — install dependencies with uv (builder)
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV TZ=Asia/Shanghai

WORKDIR /app

# Avoid writing .pyc files inside the image and use copy link mode for speed
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen \
        --no-group dev \
        --no-group torch_cpu \
        --no-group torch_gpu \
        --no-install-project

# ---------------------------------------------------------------------------
# Stage 2 — minimal runtime (python-slim, no uv, no torch)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV TZ=Asia/Shanghai

WORKDIR /app

# Copy the pre-built virtualenv and application code from the builder
COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini main.py ./
COPY config*.toml ./

ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8040

CMD ["python", "main.py"]
