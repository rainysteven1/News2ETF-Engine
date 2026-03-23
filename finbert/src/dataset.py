"""
PyTorch Dataset for labeled news parquet files.

Expected parquet columns:
  - title:     str   — news headline
  - content:   str   — news body (optional, can be empty/null)
  - level1:    str   — level-1 category label (e.g. "科技信息")
  - level2:    str   — level-2 category label (e.g. "半导体/芯片")
  - sentiment: int   — 0=negative(bearish), 1=neutral, 2=positive(bullish)
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.hierarchy import build_label_maps


def preprocess_split(
    raw_path: Path,
    data_dir: Path,
    val_ratio: float,
    seed: int = 42,
) -> None:
    """Split raw labeled data into train/val parquet files using stratified sampling.

    Skips if both output files already exist.
    """
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"

    if train_path.exists() and val_path.exists():
        return

    df = pl.read_parquet(raw_path)
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["major_category"],
        random_state=seed,
    )
    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)


class NewsClassificationDataset(Dataset):
    """Tokenized news dataset for hierarchical classification."""

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        l1_to_idx: dict[str, int] | None = None,
        l2_to_idx: dict[str, int] | None = None,
        use_content: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_content = use_content

        if l1_to_idx is None or l2_to_idx is None:
            l1_to_idx, _, l2_to_idx, _, _ = build_label_maps()
        self.l1_to_idx = l1_to_idx
        self.l2_to_idx = l2_to_idx

        df = pl.read_parquet(parquet_path)
        self._validate(df)

        self.titles = df["title"].to_list()
        self.contents = df["content"].to_list() if use_content and "content" in df.columns else None
        self.l1_labels = [self.l1_to_idx[v] for v in df["major_category"].to_list()]
        self.l2_labels = [self.l2_to_idx[v] for v in df["sub_category"].to_list()]

        # Convert string sentiment to integer labels: "利空"/"negative"→0, "中性"/"neutral"→1, "利好"/"positive"→2
        sentiment_str_to_int = {
            "利空": 0,
            "negative": 0,
            "bearish": 0,
            "中性": 1,
            "neutral": 1,
            "利好": 2,
            "positive": 2,
            "bullish": 2,
        }
        raw_sentiment = df["sentiment"].to_list()
        self.sentiment_labels = [sentiment_str_to_int[s] if isinstance(s, str) else int(s) for s in raw_sentiment]

    @staticmethod
    def _validate(df: pl.DataFrame) -> None:
        required = {"title", "major_category", "sub_category", "sentiment"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Parquet file missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.titles[idx]
        if self.contents is not None:
            content = self.contents[idx]
            if content is not None and content:
                # Prepend title to first 256 chars of content
                text = text + "[SEP]" + str(content)[:256]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = cast(torch.Tensor, encoding["input_ids"])
        attention_mask = cast(torch.Tensor, encoding["attention_mask"])
        token_type_ids = cast(torch.Tensor, encoding.get("token_type_ids", torch.zeros_like(input_ids)))

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
            "l1_label": torch.tensor(self.l1_labels[idx], dtype=torch.long),
            "l2_label": torch.tensor(self.l2_labels[idx], dtype=torch.long),
            "sentiment_label": torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
        }


class NewsInferenceDataset(Dataset):
    """Unlabeled dataset for batch inference."""

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        use_content: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pl.read_parquet(parquet_path)
        if "title" not in df.columns:
            raise ValueError("Parquet file must contain a 'title' column")

        self.titles = df["title"].to_list()
        self.contents = df["content"].to_list() if use_content and "content" in df.columns else None
        self.meta = df  # keep full df for output merging

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.titles[idx]
        if self.contents is not None:
            content = self.contents[idx]
            if content is not None and content:
                text = text + "[SEP]" + str(content)[:256]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = cast(torch.Tensor, encoding["input_ids"])
        attention_mask = cast(torch.Tensor, encoding["attention_mask"])
        token_type_ids = cast(torch.Tensor, encoding.get("token_type_ids", torch.zeros_like(input_ids)))

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
        }
