set dotenv-load := true
set dotenv-filename := ".env"

root := justfile_directory()
image := "news2etf-engine:latest"

# ── App ───────────────────────────────────────────────────────────────────────

# Start all dependency services locally (postgres / loki / grafana etc.)
local-deploy:
    docker compose up -d

# Build Docker image
build:
    docker build --network host -t {{image}} .

# Run production container
run:
    docker compose up -d --force-recreate app

# Start local dev server with hot reload
dev:
    uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8040 --reload

# ── Database migrations (Alembic) ─────────────────────────────────────────────

# Generate a migration file. Usage: just migrate "add xxx column"
migrate msg="auto":
    uv run alembic upgrade head
    uv run alembic revision --autogenerate -m "{{msg}}"

# Apply all pending migrations
db-upgrade:
    uv run alembic upgrade head

# Roll back one migration
db-downgrade:
    uv run alembic downgrade -1

# Show current migration version
db-current:
    uv run alembic current

# Show full migration history
db-history:
    uv run alembic history --verbose