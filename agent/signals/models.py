"""ML/DL model definitions: LSTM, IsolationForest, LightGBM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest as SKFIsolationForest
from sklearn.preprocessing import StandardScaler


class LSTMModel(nn.Module):
    """LSTM for sentiment momentum prediction.

    Input: (batch, seq_len, 1) — industry sentiment time series
    Output: (batch, 1) — momentum score in [-1, 1]
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.tanh(out)


class IsolationForestModel:
    """Isolation Forest for news heat anomaly detection."""

    model: SKFIsolationForest
    scaler: StandardScaler

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
    ):
        self.model = SKFIsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._fitted: bool = False

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        """Fit the model on training data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly score in [0, 1] (0 = normal, 1 = anomaly)."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        # sklearn: -1 for anomaly, 1 for normal → convert to [0, 1]
        raw = self.model.decision_function(X_scaled)
        # Normalize to [0, 1] where higher = more anomalous
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        return scores


class LightGBMModel:
    """LightGBM for composite signal prediction."""

    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
    ):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.model = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        """Fit LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("lightgbm is required for LightGBMModel") from e

        self.model = lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            verbose=-1,
        )
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return composite score in [-1, 1]."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before saving")
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> "LightGBMModel":
        """Load model from disk."""
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._fitted = True
        return self
