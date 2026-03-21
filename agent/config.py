"""Agent configuration — Pydantic models loaded from agent/config.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"


class LSTMConfig(BaseModel):
    sequence_length: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2


class IsolationForestConfig(BaseModel):
    contamination: float = 0.1
    n_estimators: int = 100


class LightGBMConfig(BaseModel):
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 100


class ModelConfig(BaseModel):
    lstm: LSTMConfig = LSTMConfig()
    isolation_forest: IsolationForestConfig = IsolationForestConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()


class AgentConfig(BaseModel):
    llm_model: str = "glm-4-flash"
    llm_temperature: float = 0.0
    max_weight_per_industry: float = 0.3
    max_total_weight: float = 1.0


class BacktestConfig(BaseModel):
    initial_capital: float = 1000000.0
    transaction_fee: float = 0.0003
    slippage: float = 0.0005
    risk_free_rate: float = 0.03


class DataConfig(BaseModel):
    input_classified: Path = _ROOT / "data" / "classified_sub.parquet"
    industry_dict: Path = _ROOT / "data" / "industry_dict.json"
    etf_prices: Path = _ROOT / "data" / "etf_prices.parquet"
    output_sentiment: Path = _ROOT / "data" / "industry_sentiment.parquet"
    output_signals: Path = _ROOT / "data" / "ml_signals.parquet"
    output_trades: Path = _ROOT / "data" / "trade_signals.parquet"
    output_logs: Path = _ROOT / "data" / "decision_logs.jsonl"
    output_backtest: Path = _ROOT / "data" / "backtest_results.parquet"
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"


class AgentRootConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    agent: AgentConfig = AgentConfig()
    backtest: BacktestConfig = BacktestConfig()
    data: DataConfig = DataConfig()


def load_config(path: Path = _CONFIG_PATH) -> AgentRootConfig:
    """Load config.toml and resolve relative paths against the project root."""
    with open(path, "rb") as f:
        raw: dict = tomllib.load(f)

    # Resolve relative paths to absolute, anchored at project root
    data_section = raw.get("data", {})
    for key in (
        "input_classified",
        "industry_dict",
        "etf_prices",
        "output_sentiment",
        "output_signals",
        "output_trades",
        "output_logs",
        "output_backtest",
    ):
        if key in data_section:
            raw["data"][key] = str(_ROOT / data_section[key])

    return AgentRootConfig.model_validate(raw)
