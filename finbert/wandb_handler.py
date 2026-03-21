"""
W&B handler for FinBERT training — standalone copy adapted from src/utils/wandb_handler.py.

Differences from the original:
  - No dependency on src.config or src.utils; accepts FinBERTConfig directly.
  - Run name is generated from timestamp + pretrained model slug.
  - Simplified to classification-training use-case (no regression metrics).
"""

from __future__ import annotations

import os
import time

import numpy as np

import wandb
from finbert.config import FinBERTConfig


def _make_run_name(cfg: FinBERTConfig) -> str:
    """Generate a short, human-readable run name."""
    model_slug = cfg.model.pretrained_model.split("/")[-1]
    ts = time.strftime("%m%d_%H%M")
    return f"finbert-{model_slug}-{ts}"


class WandbHandler:
    """Thin W&B wrapper for FinBERT training runs."""

    def __init__(self, cfg: FinBERTConfig, run_name: str | None = None):
        self.cfg = cfg
        self.run_name = run_name or _make_run_name(cfg)
        self.run = None
        self.run_id: str | None = None
        self._login()

    # ── Auth ───────────────────────────────────────────────────────────

    def _login(self) -> None:
        """Login to W&B; raises if online mode but no API key is set."""
        api_key = os.getenv("WANDB_API_KEY")
        wcfg = self.cfg.wandb
        if not api_key and wcfg.mode == "online":
            raise ValueError(
                "W&B API key not found. Set the WANDB_API_KEY environment variable "
                "or switch wandb.mode to 'offline' in config.toml."
            )
        wandb.login(key=api_key)

    # ── Run lifecycle ──────────────────────────────────────────────────

    def init_run(self) -> None:
        """Initialize a W&B run using the current FinBERTConfig."""
        cfg = self.cfg
        wcfg = cfg.wandb
        config_dict = {
            "pretrained_model": cfg.model.pretrained_model,
            "max_seq_length": cfg.model.max_seq_length,
            "dropout": cfg.model.dropout,
            "num_level1": cfg.hierarchy.num_level1,
            "num_level2": cfg.hierarchy.num_level2,
            "alpha": cfg.loss.alpha,
            "beta": cfg.loss.beta,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "warmup_ratio": cfg.training.warmup_ratio,
            "num_epochs": cfg.training.num_epochs,
            "grad_accum_steps": cfg.training.grad_accum_steps,
            "fp16": cfg.training.fp16,
            "seed": cfg.training.seed,
        }
        self.run = wandb.init(
            project=wcfg.project,
            entity=wcfg.entity or None,
            name=self.run_name,
            config=config_dict,
            mode=wcfg.mode,  # type: ignore
        )
        self.run_id = self.run.id if self.run else None

    def finish(self) -> None:
        """Mark the W&B run as finished."""
        if self.run:
            wandb.finish()

    # ── Logging ───────────────────────────────────────────────────────

    def log_metrics(self, metrics: dict[str, object], step: int | None = None) -> None:
        """Log a dict of scalars to the current run."""
        if self.run:
            wandb.log(metrics, step=step)

    def log_summary(self, metrics: dict[str, object]) -> None:
        """Write final scalars to the run summary (e.g. best_val_l2_accuracy)."""
        if self.run:
            for key, value in metrics.items():
                self.run.summary[key] = value

    def log_confusion_matrix(
        self,
        y_true: list[int],
        y_pred: list[int],
        class_names: list[str],
        title: str = "Confusion Matrix",
    ) -> None:
        """Log a normalized confusion matrix as a W&B image."""
        if not self.run:
            return

        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)  # type: ignore
        ax.figure.colorbar(im, ax=ax, format="%.2f")
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            title=f"{title} (normalized by true class)",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm_norm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        wandb.log({title.lower().replace(" ", "_"): wandb.Image(fig)})
        plt.close(fig)

    def log_artifact(
        self,
        file_path: str,
        name: str,
        artifact_type: str = "model",
        metadata: dict | None = None,
    ) -> None:
        """Upload a local file as a W&B artifact."""
        if not self.run:
            return
        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def id(self) -> str | None:
        """W&B run ID."""
        return self.run_id
