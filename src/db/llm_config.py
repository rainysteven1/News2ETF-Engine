"""LLM configuration management - loads credentials from YAML and bootstraps them to DB."""

from __future__ import annotations

import yaml
from loguru import logger

from src.common.config import CONFIGS_DIR
from src.db.models import ProviderCredential
from src.db.session import get_session


def load_yaml_credentials() -> dict:
    """Load LLM credentials from configs/llm_credentials.yaml.

    Returns:
        Dict mapping provider keys to credential dicts containing api_key and base_url.
    """
    yaml_path = CONFIGS_DIR / "llm_credentials.yaml"
    if not yaml_path.exists():
        logger.warning(f"LLM credentials file not found: {yaml_path}")
        return {}

    with open(
        yaml_path,
    ) as f:
        return yaml.safe_load(f) or {}


def bootstrap_llm_config(session) -> None:
    """Upsert provider credentials and model metadata from YAML into DB.

    Args:
        session: SQLAlchemy session (caller manages commit/rollback).
    """
    yaml_creds = load_yaml_credentials()

    # Bootstrap provider_credentials
    for provider_key, creds in yaml_creds.items():
        if not isinstance(creds, dict):
            logger.warning(f"Skipping invalid credential entry for provider: {provider_key}")
            continue
        existing = session.query(ProviderCredential).filter_by(provider_key=provider_key).first()
        if existing:
            existing.yaml_key = provider_key
            existing.api_key = creds.get("api_key", "")
            existing.base_url = creds.get("base_url", "")
            existing.is_active = True
            logger.info(f"Updated credentials for provider: {provider_key}")
        else:
            new_cred = ProviderCredential(
                provider_key=provider_key,
                yaml_key=provider_key,
                api_key=creds.get("api_key", ""),
                base_url=creds.get("base_url", ""),
                is_active=True,
            )
            session.add(new_cred)
            logger.info(f"Inserted credentials for provider: {provider_key}")

    session.commit()
    logger.info("LLM credentials bootstrap complete")


def ensure_llm_config() -> None:
    """Application startup entry point - initializes DB and bootstraps LLM config.

    Creates tables if needed, then loads YAML credentials into DB.
    """
    from src.db.session import init_db

    init_db()
    with get_session() as session:
        bootstrap_llm_config(session)
