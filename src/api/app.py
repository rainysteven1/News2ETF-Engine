"""FastAPI application factory."""

import json
import os
from contextlib import asynccontextmanager
from typing import Any

import requests
from alembic.config import Config
from fastapi import FastAPI
from loguru import logger

from alembic import command
from src.api.routers import data, experiments, industry, labeling, tasks
from src.common.config import ROOT_DIR

# Controlled by main.py CLI; default True so `uvicorn src.api.app:app` still migrates.
# Set RUN_MIGRATE=false via env var to skip migrations (e.g. for swagger export).
RUN_MIGRATE: bool = os.getenv("RUN_MIGRATE", "true").lower() != "false"
SYNC_APIFOX: bool = False


def _sync_to_apifox(schema: dict[str, Any]) -> None:
    """Push the OpenAPI schema to an Apifox project via the import-openapi API."""
    url = f"https://api.apifox.com/v1/projects/{os.getenv('APIFOX_PROJECT_ID', '')}/import-openapi?locale=zh-CN"
    payload = json.dumps(
        {
            "input": {"content": json.dumps(schema)},
            "options": {},
        }
    )
    headers = {
        "X-Apifox-Api-Version": "2024-03-28",
        "Authorization": f"Bearer {os.getenv('APIFOX_API_KEY', '')}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, data=payload)
    logger.info(f"Apifox sync: {response.status_code} {response.text}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if RUN_MIGRATE:
        alembic_cfg = Config(str(ROOT_DIR / "alembic.ini"))
        command.upgrade(alembic_cfg, "head")
    if SYNC_APIFOX:
        logger.info("Syncing OpenAPI schema to Apifox...")
        try:
            _sync_to_apifox(app.openapi())
        except requests.exceptions.ConnectionError:
            logger.warning("Apifox sync skipped: cannot reach api.apifox.com (no external network?)")
        except Exception as e:
            logger.warning(f"Apifox sync failed: {e}")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="News2ETF-Engine",
        description="News labeling and experiment management API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
    app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
    app.include_router(labeling.router, prefix="/labeling", tags=["labeling"])
    app.include_router(data.router, prefix="/data", tags=["data"])
    app.include_router(industry.router, prefix="/industry", tags=["industry"])

    @app.get("/", tags=["health"])
    def health() -> dict:
        return {"status": "ok", "service": "News2ETF-Engine"}

    return app


# Module-level instance for uvicorn string reference ("src.api.app:app")
app = create_app()
