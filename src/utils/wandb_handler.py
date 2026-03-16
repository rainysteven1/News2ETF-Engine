"""
W&B Handler module - Wrapper for W&B operations
"""

import os

import numpy as np

import wandb
from src.config import AppConfig
from src.utils.id_gen import generate_wandb_run_name


class WandbHandler:
    """W&B Handler"""

    def __init__(self, config: AppConfig, experiment_id: str):
        """
        Initialize W&B Handler

        Args:
            config: Configuration object
            experiment_id: Experiment ID
        """
        self.config = config
        self.experiment_id = experiment_id
        self.run = None
        self.run_id = None

        self._login()

    def _login(self):
        """Login to W&B using API key from environment variable"""
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key and self.config.wandb.mode == "online":
            raise ValueError("W&B API key not found in environment variable 'WANDB_API_KEY'")
        wandb.login(key=api_key)

    def init_run(self):
        """Initialize wandb run"""
        exp_fields = {
            "model_name",
            "batch_size",
            "epochs",
            "learning_rate",
            "hidden_dims",
            "optimizer",
            "scheduler_type",
            "val_ratio",
        }
        config_dict = {
            "experiment_id": self.experiment_id,
            **{k: v for k, v in self.config.model_dump(include=exp_fields).items() if v is not None},
        }

        self.run = wandb.init(
            project=self.config.project_name,
            entity=self.config.entity or None,
            name=generate_wandb_run_name(self.experiment_id),
            config=config_dict,
            mode=self.config.wandb.mode,  # type: ignore
        )
        self.run_id = self.run.id if self.run else None

    def log_metrics(self, metrics: dict[str, object], step: int | None = None):
        """
        Log metrics

        Args:
            metrics: Metric dictionary
            step: Step number
        """
        if self.run:
            wandb.log(metrics, step=step)

    def log_model_params(self, model):
        """
        Log model parameter statistics

        Args:
            model: Model object
        """
        if self.run and self.config.wandb.log_parameters:
            if hasattr(model, "get_params"):
                params = model.get_params()
                self.log_metrics({"model_params": params})

    def log_summary(self, metrics: dict[str, object]):
        """Log summary metrics (e.g. best_val_accuracy) to W&B run summary."""
        if self.run:
            for key, value in metrics.items():
                self.run.summary[key] = value

    def log_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Log confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
        """
        if not self.run:
            return

        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        # Normalize per true class (row) → recall matrix
        cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)  # type: ignore
        ax.figure.colorbar(im, ax=ax, format="%.2f")

        # Set ticks
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            title="Confusion Matrix (Normalized by True Class)",
            ylabel="True Label",
            xlabel="Predicted Label",
        )

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add values: show normalized ratio + raw count
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

        # Log to W&B
        wandb.log({"confusion_matrix": wandb.Image(fig)})

        plt.close(fig)

    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model", metadata: dict | None = None):
        """
        Upload a file as a W&B artifact.

        Args:
            file_path: Local path to the file to upload
            name: Artifact name
            artifact_type: Artifact type (default: "model")
            metadata: Optional metadata dict attached to the artifact
        """
        if not self.run:
            return
        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

    def finish(self):
        """Finish current run"""
        if self.run:
            wandb.finish()

    @property
    def id(self) -> str | None:
        """Get W&B run ID"""
        return self.run_id
