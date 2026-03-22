import os
import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, computed_field
from rich.console import Console

console = Console()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"


# ── Pydantic config models ────────────────────────────────────────────────────

_DRIVER_MAP: dict[str, str] = {
    "postgresql": "postgresql+psycopg2",
    "mysql": "mysql+pymysql",
    "sqlite": "sqlite",
}


class DatabaseConfig(BaseModel):
    type: str = "postgresql"
    ip: str = "localhost"
    port: int = 5432
    user: str = "news2etf"
    password: str = "news2etf_password"
    db: str = "news2etf"

    @computed_field
    @property
    def url(self) -> str:
        """Build SQLAlchemy connection URL, resolving driver from type."""
        driver = _DRIVER_MAP.get(self.type, self.type)
        return f"{driver}://{self.user}:{self.password}@{self.ip}:{self.port}/{self.db}"


class LokiConfig(BaseModel):
    url: str = "http://localhost:3100"


class S3Config(BaseModel):
    endpoint: str = "http://localhost:8333"
    access_key: str = ""
    secret_key: str = ""


class LabelingConfig(BaseModel):
    batch_size: int = 20
    checkpoint_every: int = 5
    s3_bucket: str = "labeling"
    llm_retry: int = 2


class ClickHouseConfig(BaseModel):
    host: str = "localhost"
    port: int = 9000
    database: str = "default"
    user: str = "default"
    password: str = ""


class AppConfig(BaseModel):
    database: DatabaseConfig = DatabaseConfig()
    loki: LokiConfig = LokiConfig()
    s3: S3Config = S3Config()
    labeling: LabelingConfig = LabelingConfig()
    clickhouse: ClickHouseConfig = ClickHouseConfig()


# ── Loader ────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load config file selected by ENV_MODE env var.

    Resolution order:
      1. config.{ENV_MODE}.toml  (e.g. config.docker.toml, config.dev.toml)
      2. config.toml             (fallback)
      3. Pydantic defaults       (if no file found)
    """
    env_mode = os.environ.get("ENV_MODE", "")
    candidates = []
    if env_mode:
        candidates.append(ROOT_DIR / f"config.{env_mode}.toml")
    candidates.append(ROOT_DIR / "config.toml")

    config_path = next((p for p in candidates if p.exists()), None)
    raw: dict = {}
    if config_path is not None:
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
    return AppConfig.model_validate(raw)
