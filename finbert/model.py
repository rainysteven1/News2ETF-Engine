"""
FinBERT with multi-head hierarchical classifier.

Architecture:
  ┌────────────────────────┐
  │  BERT Backbone (shared) │
  │  [CLS] pooled output   │
  └──────────┬─────────────┘
             │
     ┌───────┴───────┐
     │               │
  ┌──▼──┐        ┌───▼──┐
  │ L1   │        │ L2   │
  │ Head │        │ Head │
  │ (8)  │        │ (28) │
  └──────┘        └──────┘

Joint loss: L = α × CE(L1) + β × CE(L2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, PretrainedConfig


class HierarchicalClassifierConfig(PretrainedConfig):
    model_type = "hierarchical_classifier"

    def __init__(
        self,
        num_level1: int = 8,
        num_level2: int = 28,
        num_sentiment: int = 3,
        classifier_dropout: float = 0.1,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_level1 = num_level1
        self.num_level2 = num_level2
        self.num_sentiment = num_sentiment
        self.classifier_dropout = classifier_dropout
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class FinBERTHierarchicalClassifier(BertPreTrainedModel):
    """BERT backbone + two classification heads for hierarchical labeling."""

    config_class = HierarchicalClassifierConfig

    def __init__(self, config: HierarchicalClassifierConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        hidden = config.hidden_size
        drop = config.classifier_dropout

        self.dropout = nn.Dropout(drop)

        # Level-1 head: 8 major categories
        self.l1_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 2, config.num_level1),
        )

        # Level-2 head: 28 subcategories (global index)
        self.l2_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 2, config.num_level2),
        )

        # Sentiment head: negative(0) / neutral(1) / positive(2)
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 4, config.num_sentiment),
        )

        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.loss_fn = nn.CrossEntropyLoss()

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        l1_label: torch.Tensor | None = None,
        l2_label: torch.Tensor | None = None,
        sentiment_label: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self.dropout(outputs.pooler_output)  # (B, H)

        l1_logits = self.l1_head(pooled)  # (B, num_l1)
        l2_logits = self.l2_head(pooled)  # (B, num_l2)
        sent_logits = self.sentiment_head(pooled)  # (B, num_sentiment)

        result: dict[str, torch.Tensor] = {
            "l1_logits": l1_logits,
            "l2_logits": l2_logits,
            "sentiment_logits": sent_logits,
        }

        if l1_label is not None and l2_label is not None and sentiment_label is not None:
            l1_loss = self.loss_fn(l1_logits, l1_label)
            l2_loss = self.loss_fn(l2_logits, l2_label)
            sent_loss = self.loss_fn(sent_logits, sentiment_label)
            result["loss"] = self.alpha * l1_loss + self.beta * l2_loss + self.gamma * sent_loss
            result["l1_loss"] = l1_loss
            result["l2_loss"] = l2_loss
            result["sentiment_loss"] = sent_loss

        return result


def load_finbert_classifier(
    pretrained_model: str,
    num_level1: int = 8,
    num_level2: int = 28,
    num_sentiment: int = 3,
    dropout: float = 0.1,
    alpha: float = 0.3,
    beta: float = 0.5,
    gamma: float = 0.2,
) -> FinBERTHierarchicalClassifier:
    """Initialize hierarchical classifier from a pretrained BERT checkpoint."""
    config = HierarchicalClassifierConfig.from_pretrained(
        pretrained_model,
        num_level1=num_level1,
        num_level2=num_level2,
        num_sentiment=num_sentiment,
        classifier_dropout=dropout,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    model = FinBERTHierarchicalClassifier.from_pretrained(
        pretrained_model,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model
