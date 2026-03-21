"""
FinBERT configuration — Pydantic models loaded from finbert/config.toml.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"


class ModelConfig(BaseModel):
    pretrained_model: str = "IDEA-CCNL/Erlangshen-FinBERT-base"
    max_seq_length: int = 128
    dropout: float = 0.1


class HierarchyConfig(BaseModel):
    num_level1: int = 8
    num_level2: int = 28
    num_sentiment: int = 3  # negative(0) / neutral(1) / positive(2)


class LossConfig(BaseModel):
    alpha: float = 0.3  # L1 loss weight
    beta: float = 0.5  # L2 loss weight
    gamma: float = 0.2  # sentiment loss weight


class TrainingConfig(BaseModel):
    batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 5
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    fp16: bool = True


class DataConfig(BaseModel):
    data_dir: Path = _ROOT / "data" / "labeled"
    train_file: str = "train.parquet"
    val_file: str = "val.parquet"
    test_file: str = "test.parquet"


class OutputConfig(BaseModel):
    output_dir: Path = _ROOT / "finbert" / "checkpoints"
    log_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500


class InferenceConfig(BaseModel):
    checkpoint_dir: Path = _ROOT / "finbert" / "checkpoints" / "best"
    input_file: Path = _ROOT / "data" / "converted" / "tushare_news_2021_today_part1.parquet"
    output_file: Path = _ROOT / "data" / "classified.parquet"
    infer_batch_size: int = 256
    use_content: bool = False


class WandbConfig(BaseModel):
    project: str = "news2etf-finbert"
    entity: str = ""  # W&B team / username; empty = personal account
    mode: str = "offline"  # online | offline | disabled
    log_l2_cm: bool = False  # log level-2 confusion matrix at end of training


class FinBERTConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    hierarchy: HierarchyConfig = HierarchyConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    output: OutputConfig = OutputConfig()
    inference: InferenceConfig = InferenceConfig()
    wandb: WandbConfig = WandbConfig()


def load_config(path: Path = _CONFIG_PATH) -> FinBERTConfig:
    """Load config.toml and resolve relative paths against the project root."""
    with open(path, "rb") as f:
        raw: dict = tomllib.load(f)

    # Resolve relative paths to absolute, anchored at project root
    if "data" in raw and "data_dir" in raw["data"]:
        raw["data"]["data_dir"] = _ROOT / raw["data"]["data_dir"]
    if "output" in raw and "output_dir" in raw["output"]:
        raw["output"]["output_dir"] = _ROOT / raw["output"]["output_dir"]
    if "inference" in raw:
        for key in ("checkpoint_dir", "input_file", "output_file"):
            if key in raw["inference"]:
                raw["inference"][key] = _ROOT / raw["inference"][key]

    return FinBERTConfig.model_validate(raw)
