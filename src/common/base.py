import tomllib
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"


def load_config() -> dict[str, Any]:
    """Load config.toml from project root. Returns empty dict if missing."""
    config_path = ROOT_DIR / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return {}


def get_loki_url() -> str | None:
    """Return the Loki push URL. Env var LOKI_URL takes precedence over config.toml."""
    import os

    env_url = os.environ.get("LOKI_URL")
    if env_url:
        return env_url
    cfg = load_config()
    return cfg.get("loki", {}).get("url")


def get_labeling_config(level: int) -> dict[str, Any]:
    """Return LLM params for the given labeling level (1 or 2) from config.toml.

    Defaults match the original hardcoded values so existing deployments
    without the new config keys continue to work unchanged.
    """
    defaults: dict[int, dict[str, Any]] = {
        1: {"model": "glm-4.5-airx", "temperature": 0.1, "max_tokens": 8192},
        2: {"model": "glm-4.7-flashx", "temperature": 0.1, "max_tokens": 4096},
    }
    cfg = load_config()
    section = cfg.get("labeling", {}).get(f"level{level}", {})
    base = defaults.get(level, {})
    return {**base, **section}
