"""
Batch inference pipeline for FinBERT hierarchical classifier.

Loads a trained checkpoint and runs prediction on unlabeled parquet files.
Applies hierarchical masking: L2 predictions are constrained to subcategories
of the predicted L1 class.  Outputs (news, level1, level2, sentiment) tuples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from finbert.dataset import NewsInferenceDataset
from finbert.hierarchy import build_l1_to_l2_indices, build_label_maps
from finbert.model import FinBERTHierarchicalClassifier

SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def predict(
    checkpoint_dir: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    batch_size: int = 256,
    max_length: int = 128,
    use_content: bool = False,
) -> pl.DataFrame:
    """Run hierarchical classification on an unlabeled parquet file.

    Returns a Polars DataFrame with added columns:
    level1, level2, sentiment, l1_confidence, l2_confidence, sentiment_confidence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = Path(checkpoint_dir)

    # ── Load model & tokenizer ─────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = FinBERTHierarchicalClassifier.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    # ── Label maps ─────────────────────────────────────
    _, idx_to_l1, _, idx_to_l2, _ = build_label_maps()
    l1_to_l2_indices = build_l1_to_l2_indices()

    # ── Build L2 mask matrix: (num_l1, num_l2) ────────
    num_l1 = len(idx_to_l1)
    num_l2 = len(idx_to_l2)
    l2_mask = torch.full((num_l1, num_l2), float("-inf"), device=device)
    for l1_idx, l2_ids in l1_to_l2_indices.items():
        for l2_idx in l2_ids:
            l2_mask[l1_idx, l2_idx] = 0.0

    # ── Dataset & loader ───────────────────────────────
    dataset = NewsInferenceDataset(input_path, tokenizer, max_length=max_length, use_content=use_content)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_l1_preds: list[int] = []
    all_l2_preds: list[int] = []
    all_l1_confs: list[float] = []
    all_l2_confs: list[float] = []
    all_sent_preds: list[int] = []
    all_sent_confs: list[float] = []

    print(f"Running inference on {len(dataset)} samples...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )

            # L1 prediction
            l1_probs = torch.softmax(outputs["l1_logits"], dim=-1)
            l1_conf, l1_pred = l1_probs.max(dim=-1)

            # L2 prediction with hierarchical masking
            batch_l2_mask = l2_mask[l1_pred]  # (B, num_l2)
            masked_l2_logits = outputs["l2_logits"] + batch_l2_mask
            l2_probs = torch.softmax(masked_l2_logits, dim=-1)
            l2_conf, l2_pred = l2_probs.max(dim=-1)

            # Sentiment prediction
            sent_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
            sent_conf, sent_pred = sent_probs.max(dim=-1)

            all_l1_preds.extend(l1_pred.cpu().tolist())
            all_l2_preds.extend(l2_pred.cpu().tolist())
            all_l1_confs.extend(l1_conf.cpu().tolist())
            all_l2_confs.extend(l2_conf.cpu().tolist())
            all_sent_preds.extend(sent_pred.cpu().tolist())
            all_sent_confs.extend(sent_conf.cpu().tolist())

            if (i + 1) % 50 == 0:
                print(f"  Processed {(i + 1) * batch_size}/{len(dataset)} samples")

    # ── Assemble results (Polars) ───────────────────────
    df = dataset.meta.with_columns(
        pl.Series("level1", [idx_to_l1[i] for i in all_l1_preds]),
        pl.Series("level2", [idx_to_l2[i] for i in all_l2_preds]),
        pl.Series("sentiment", [SENTIMENT_LABELS[i] for i in all_sent_preds]),
        pl.Series("l1_confidence", np.round(all_l1_confs, 4).tolist()),
        pl.Series("l2_confidence", np.round(all_l2_confs, 4).tolist()),
        pl.Series("sentiment_confidence", np.round(all_sent_confs, 4).tolist()),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"Results saved to {output_path} ({len(df)} rows)")

    return df
