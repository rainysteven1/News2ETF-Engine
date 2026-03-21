"""LLM client factory.

Reads provider credentials from environment variables and returns an
OpenAI-compatible client.  The provider is looked up from an explicit
model registry — MODEL_REGISTRY maps exact model names to providers.

To add a new model: add its exact name to MODEL_REGISTRY below.
To add a new provider: add a ProviderConfig entry to PROVIDERS.
"""

from __future__ import annotations

import os

from openai import OpenAI
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    key_env: str
    url_env: str
    default_url: str

    def api_key(self) -> str | None:
        return os.environ.get(self.key_env)

    def base_url(self) -> str:
        return os.environ.get(self.url_env) or self.default_url


PROVIDERS: dict[str, ProviderConfig] = {
    "zhipu": ProviderConfig(
        key_env="ZHIPU_API_KEY",
        url_env="ZHIPU_BASE_URL",
        default_url="https://open.bigmodel.cn/api/paas/v4",
    ),
    "minimax": ProviderConfig(
        key_env="MINIMAX_API_KEY",
        url_env="MINIMAX_BASE_URL",
        default_url="https://api.minimaxi.chat/v1",
    ),
    "openai": ProviderConfig(
        key_env="OPENAI_API_KEY",
        url_env="OPENAI_BASE_URL",
        default_url="https://api.openai.com/v1",
    ),
}

# Explicit model registry — exact model name → provider key in PROVIDERS
# Add new models here as they become available.
MODEL_REGISTRY: dict[str, str] = {
    # ── Zhipu GLM ──────────────────────────────────────────────────────────
    "glm-4-flash": "zhipu",
    "glm-4.7": "zhipu",
    "glm-4.5-airx": "zhipu",
    "glm-4.7-flashx": "zhipu",
    # ── MiniMax ────────────────────────────────────────────────────────────
    "MiniMax-M2.7": "minimax",
    "MiniMax-M2.7-highspeed": "minimax",
    "MiniMax-M2.5": "minimax",
    "MiniMax-M2.5-highspeed": "minimax",
    "MiniMax-M2.1": "minimax",
    "MiniMax-M2.1-highspeed": "minimax",
    # ── OpenAI ─────────────────────────────────────────────────────────────
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4-turbo": "openai",
    "o1": "openai",
    "o1-mini": "openai",
    "o3": "openai",
    "o3-mini": "openai",
}


def resolve_provider(model: str) -> ProviderConfig:
    """Return the ProviderConfig for *model*.

    Raises ValueError if the model is not in MODEL_REGISTRY.
    """
    provider_key = MODEL_REGISTRY.get(model)
    if provider_key is None:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{model}'. Add it to MODEL_REGISTRY in agent/utils/llm_client.py. Known models: {known}"
        )
    return PROVIDERS[provider_key]


def get_llm_client(model: str) -> OpenAI:
    """Return an OpenAI-compatible client for *model*.

    Raises:
        ValueError: If model is unknown or the required API key env var is not set.
    """
    provider = resolve_provider(model)
    api_key = provider.api_key()
    if not api_key:
        raise ValueError(f"Model '{model}' requires env var {provider.key_env} to be set")
    return OpenAI(api_key=api_key, base_url=provider.base_url())
