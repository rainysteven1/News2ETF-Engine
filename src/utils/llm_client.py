"""LLM client factory.

Reads provider credentials and model registry from PostgreSQL DB (bootstrapped from
configs/llm_credentials.yaml) and returns an OpenAI-compatible client.
"""

from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Provider configuration loaded from DB."""

    provider_key: str
    api_key: str
    base_url: str


# In-memory caches: loaded from DB at startup
_PROVIDER_CACHE: dict[str, ProviderConfig] = {}
_MODEL_REGISTRY: dict[str, str] = {}  # model_name -> provider_key


def _load_from_db() -> None:
    """Load provider credentials and model metadata from DB into memory cache.

    Called at module import time to ensure credentials are available.
    """
    global _PROVIDER_CACHE, _MODEL_REGISTRY

    try:
        from src.db.models import ModelMetaRule, ProviderCredential
        from src.db.session import get_session

        with get_session() as session:
            # Load active provider credentials
            creds = session.query(ProviderCredential).filter_by(is_active=True).all()
            for cred in creds:
                _PROVIDER_CACHE[cred.provider_key] = ProviderConfig(
                    provider_key=cred.provider_key,
                    api_key=cred.api_key,
                    base_url=cred.base_url,
                )

            # Load model metadata
            models = session.query(ModelMetaRule).all()
            _MODEL_REGISTRY = {m.model_name: m.provider_key for m in models}

    except Exception:
        # If DB is not available yet (e.g., import before DB init), leave cache empty
        pass


def resolve_provider(model: str) -> ProviderConfig:
    """Return the ProviderConfig for *model*.

    Raises ValueError if the model is not in model registry or provider is not loaded.
    """
    if not _PROVIDER_CACHE:
        _load_from_db()

    provider_key = _MODEL_REGISTRY.get(model)
    if provider_key is None:
        known = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{model}'. Add it via alembic migration. Known models: {known}"
        )
    provider = _PROVIDER_CACHE.get(provider_key)
    if provider is None:
        raise ValueError(f"Provider '{provider_key}' for model '{model}' is not loaded in credentials cache")
    return provider


def get_llm_client(model: str) -> OpenAI:
    """Return an OpenAI-compatible client for *model*.

    Raises:
        ValueError: If model is unknown or the provider credentials are not available.
    """
    provider = resolve_provider(model)
    if not provider.api_key:
        raise ValueError(f"Model '{model}' has no API key configured for provider '{provider.provider_key}'")
    return OpenAI(api_key=provider.api_key, base_url=provider.base_url)


# Load credentials at module import
_load_from_db()
