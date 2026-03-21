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

import polars as pl
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from finbert.hierarchy import build_label_maps


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
        self.l1_labels = [self.l1_to_idx[v] for v in df["level1"].to_list()]
        self.l2_labels = [self.l2_to_idx[v] for v in df["level2"].to_list()]
        self.sentiment_labels = df["sentiment"].to_list()

    @staticmethod
    def _validate(df: pl.DataFrame) -> None:
        required = {"title", "level1", "level2", "sentiment"}
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

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
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

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
        }
