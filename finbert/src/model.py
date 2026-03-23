"""
FinBERT with multi-head hierarchical classifier.

Architecture:
  ┌─────────────────────────────────┐
  │  BERT Backbone (shared)          │
  │  [CLS] pooled output            │
  │  last_hidden_state (all tokens)  │
  └──────────┬──────────────────────┘
             │
       ┌─────┴─────┐
       │           │
  ┌────▼───┐   ┌───▼────────────┐
  │ CLS    │   │ mean_pool       │
  │ (sent) │   │ (L1 + L2)       │
  └────────┘   └────────────────┘
                     │
                ┌────▼────┐
                │ L1 feat │──────► L2_head (hierarchical)
                └─────────┘

Joint loss: L = α × CE(L1) + β × CE(L2) + γ × CE(sent)

Weight initialization:
  - Hidden layers (Linear + GELU): Kaiming normal (fan_out, relu)
  - Output layers (final Linear): Xavier uniform
  - L2 input: LayerNorm after concat to normalize pretrained + random features
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


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
        bert_lr: float = 2e-5,
        heads_lr: float = 1e-4,
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
        self.bert_lr = bert_lr
        self.heads_lr = heads_lr


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over non-padded tokens.

    Args:
        hidden_states: (B, L, H)
        attention_mask: (B, L) - 1 for real tokens, 0 for padding

    Returns:
        (B, H) pooled representation
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


class FinBERTHierarchicalClassifier(BertPreTrainedModel):
    """BERT backbone + hierarchical classification heads.

    - Sentiment head: uses CLS (pooled output) - focuses on adjectives/tone
    - L1/L2 heads: use mean pooling - better for industry term classification
    - L2 head: receives L1 hidden features for hierarchical constraint
    - All heads use proper initialization (Kaiming for hidden, Xavier for output)
    """

    config_class = HierarchicalClassifierConfig

    def __init__(self, config: HierarchicalClassifierConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        hidden = config.hidden_size
        drop = config.classifier_dropout

        self.dropout = nn.Dropout(drop)

        # L1 head: separate layers for easier weight init access
        self.l1_fc1 = nn.Linear(hidden, hidden // 2)
        self.l1_activation = nn.GELU()
        self.l1_dropout = nn.Dropout(drop)
        self.l1_fc2 = nn.Linear(hidden // 2, config.num_level1)

        # L2 head: mean_pooled + L1_hidden -> LayerNorm -> fc -> fc -> out
        # LayerNorm normalizes pretrained (stable) and random-init (unstable) features
        l2_input_dim = hidden + hidden // 2
        self.l2_ln = nn.LayerNorm(l2_input_dim)
        self.l2_fc1 = nn.Linear(l2_input_dim, hidden // 2)
        self.l2_activation = nn.GELU()
        self.l2_dropout = nn.Dropout(drop)
        self.l2_fc2 = nn.Linear(hidden // 2, config.num_level2)

        # Sentiment head
        self.sent_fc1 = nn.Linear(hidden, hidden // 4)
        self.sent_activation = nn.GELU()
        self.sent_dropout = nn.Dropout(drop)
        self.sent_fc2 = nn.Linear(hidden // 4, config.num_sentiment)

        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.loss_fn = nn.CrossEntropyLoss()

        # Apply custom initialization to heads (BERT stays pretrained)
        self.apply(self._init_weights)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Custom weight initialization for classification heads.

        - Hidden layers (Linear + GELU): Kaiming normal (fan_out, nonlinearity=relu)
        - Output layers (final Linear): Xavier uniform
        - LayerNorm: default initialization (bias=0, weight=1)
        - BERT backbone: left untouched (keeps pretrained weights)
        """
        if isinstance(module, nn.Linear):
            # Determine if this is a hidden layer or output layer
            # Hidden layers feed into GELU, output layers produce logits
            if module.out_features in [self.config.hidden_size // 2, self.config.hidden_size // 4]:
                # Hidden layer: Kaiming normal for GELU
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            else:
                # Output layer: Xavier uniform for stable initial logits
                nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

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

        # Sentiment: CLS pooled output
        pooled = self.dropout(outputs.pooler_output)  # (B, H)

        # L1/L2: mean pooling over all tokens (better for industry terms)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mean_pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        mean_pooled = self.dropout(mean_pooled)  # (B, H)

        # L1 forward pass
        l1_hidden = self.l1_activation(self.l1_fc1(mean_pooled))  # (B, H//2)
        l1_hidden = self.l1_dropout(l1_hidden)
        l1_logits = self.l1_fc2(l1_hidden)  # (B, num_l1)

        # L2 forward pass: concat + LayerNorm + FC
        # LayerNorm prevents random-init L1 features from dominating pretrained mean_pooled
        l2_input = torch.cat([mean_pooled, l1_hidden], dim=-1)  # (B, H + H//2)
        l2_input_norm = self.l2_ln(l2_input)
        l2_hidden = self.l2_activation(self.l2_fc1(l2_input_norm))
        l2_hidden = self.l2_dropout(l2_hidden)
        l2_logits = self.l2_fc2(l2_hidden)  # (B, num_l2)

        # Sentiment forward pass
        sent_hidden = self.sent_activation(self.sent_fc1(pooled))
        sent_hidden = self.sent_dropout(sent_hidden)
        sent_logits = self.sent_fc2(sent_hidden)  # (B, num_sentiment)

        result: dict[str, torch.Tensor] = {
            "l1_logits": l1_logits,
            "l2_logits": l2_logits,
            "sentiment_logits": sent_logits,
        }

        # Flexible loss calculation - handle missing labels
        if l1_label is not None:
            l1_loss = self.loss_fn(l1_logits, l1_label)
            total_loss = self.alpha * l1_loss

            result["loss"] = total_loss
            result["l1_loss"] = l1_loss

            if l2_label is not None:
                l2_loss = self.loss_fn(l2_logits, l2_label)
                total_loss = total_loss + self.beta * l2_loss
                result["l2_loss"] = l2_loss

            if sentiment_label is not None:
                sent_loss = self.loss_fn(sent_logits, sentiment_label)
                total_loss = total_loss + self.gamma * sent_loss
                result["sentiment_loss"] = sent_loss

        return result

    def get_optim_params(self) -> list[dict[str, Any]]:
        return [
            {"params": self.bert.parameters(), "lr": self.config.bert_lr},
            {"params": self.l1_fc1.parameters(), "lr": self.config.heads_lr},
            {"params": self.l1_fc2.parameters(), "lr": self.config.heads_lr},
            {"params": self.l2_fc1.parameters(), "lr": self.config.heads_lr},
            {"params": self.l2_fc2.parameters(), "lr": self.config.heads_lr},
            {"params": self.l2_ln.parameters(), "lr": self.config.heads_lr},
            {"params": self.sent_fc1.parameters(), "lr": self.config.heads_lr},
            {"params": self.sent_fc2.parameters(), "lr": self.config.heads_lr},
        ]


def load_finbert_classifier(
    pretrained_model: str,
    num_level1: int = 8,
    num_level2: int = 28,
    num_sentiment: int = 3,
    dropout: float = 0.1,
    alpha: float = 0.3,
    beta: float = 0.5,
    gamma: float = 0.2,
    bert_lr: float = 2e-5,
    heads_lr: float = 1e-4,
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
        bert_lr=bert_lr,
        heads_lr=heads_lr,
    )
    model = FinBERTHierarchicalClassifier.from_pretrained(
        pretrained_model,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model  # type: ignore
