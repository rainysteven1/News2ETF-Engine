"""Signal scorer — computes daily ML signals for each industry."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from agent.config import AgentRootConfig
from agent.signals.models import LSTMModel


class SignalScorer:
    """Computes ML signals for each industry on a given date."""

    def __init__(
        self,
        config: AgentRootConfig,
        checkpoint_dir: Path,
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Load LSTM
        self.lstm = LSTMModel(
            hidden_size=config.model.lstm.hidden_size,
            num_layers=config.model.lstm.num_layers,
            dropout=config.model.lstm.dropout,
        )
        lstm_path = checkpoint_dir / "lstm_model.pt"
        if lstm_path.exists():
            self.lstm.load_state_dict(torch.load(lstm_path, weights_only=True))
            self.lstm.eval()
        else:
            raise FileNotFoundError(f"LSTM model not found at {lstm_path}")

        # Load Isolation Forest
        self.iforest: Any
        iforest_path = checkpoint_dir / "iforest_model.pkl"
        if iforest_path.exists():
            with open(iforest_path, "rb") as f:
                self.iforest = pickle.load(f)
        else:
            raise FileNotFoundError(f"Isolation Forest model not found at {iforest_path}")

        # Load LightGBM
        self.lgbm = None
        lgbm_path = checkpoint_dir / "lgbm_model.txt"
        if lgbm_path.exists():
            try:
                import lightgbm as lgb
                booster = lgb.Booster(model_file=str(lgbm_path))
                self.lgbm = booster
            except Exception as e:
                raise FileNotFoundError(f"LightGBM model load failed: {e}")
        else:
            raise FileNotFoundError(f"LightGBM model not found at {lgbm_path}")

    def _get_lookback_window(
        self,
        sentiment_df: pl.DataFrame,
        industry: str,
        date: str,
        window: int,
    ) -> np.ndarray | None:
        """Get the last `window` days of sentiment for an industry up to (and including) date."""
        ind_df = (
            sentiment_df
            .filter(pl.col("industry") == industry)
            .filter(pl.col("date") <= date)
            .sort("date")
            .tail(window)
        )
        if len(ind_df) < window:
            return None
        return ind_df["sentiment_mean"].to_numpy().astype(np.float32)

    def compute_lstm_momentum(
        self,
        sentiment_df: pl.DataFrame,
        industry: str,
        date: str,
    ) -> float:
        """Compute LSTM momentum score for an industry on a date."""
        seq_len = self.config.model.lstm.sequence_length
        seq = self._get_lookback_window(sentiment_df, industry, date, seq_len)
        if seq is None:
            return 0.0

        seq_tensor = torch.FloatTensor(seq.reshape(1, seq_len, 1))
        with torch.no_grad():
            score = self.lstm(seq_tensor).item()
        return float(score)

    def compute_iforest_anomaly(
        self,
        sentiment_df: pl.DataFrame,
        industry: str,
        date: str,
    ) -> float:
        """Compute Isolation Forest anomaly score for an industry on a date."""
        from sklearn.preprocessing import StandardScaler

        seq_len = self.config.model.lstm.sequence_length

        ind_df = (
            sentiment_df
            .filter(pl.col("industry") == industry)
            .filter(pl.col("date") <= date)
            .sort("date")
            .tail(seq_len * 2)
        )
        if len(ind_df) < seq_len * 2:
            return 0.0

        count_seq = ind_df["news_count"].to_numpy().astype(np.float32)[-seq_len:]
        heat_seq = ind_df["news_heat"].to_numpy().astype(np.float32)[-seq_len:]
        features = np.concatenate([count_seq, heat_seq]).reshape(1, -1)

        # Simple scaling based on training params
        scaler = StandardScaler()
        scaler.fit(features)
        features_scaled = scaler.transform(features)

        raw_score = self.iforest.decision_function(features_scaled)
        # Normalize to [0, 1]
        normalized = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-8)
        return float(normalized[0])

    def compute_lgbm_composite(
        self,
        sentiment_df: pl.DataFrame,
        industry: str,
        date: str,
    ) -> float:
        """Compute LightGBM composite score for an industry on a date."""
        seq_len = self.config.model.lstm.sequence_length
        ind_df = (
            sentiment_df
            .filter(pl.col("industry") == industry)
            .filter(pl.col("date") <= date)
            .sort("date")
        )
        n = len(ind_df)
        if n < seq_len + 1:
            return 0.0

        sentiment_mean = ind_df["sentiment_mean"].to_numpy()
        news_count = ind_df["news_count"].to_numpy()
        news_heat = ind_df["news_heat"].to_numpy()
        sentiment_trend = ind_df["sentiment_trend"].to_numpy()

        i = n - 1
        mom_score = sentiment_mean[i] - sentiment_mean[i - seq_len]
        heat_val = news_heat[i] if i < len(news_heat) else 0.0
        trend_val = sentiment_trend[i] if i < len(sentiment_trend) else 0.0
        count_val = news_count[i]

        features = np.array([[mom_score, heat_val, trend_val, count_val]], dtype=np.float32)
        score = self.lgbm.predict(features)[0]
        return float(np.clip(score, -1.0, 1.0))

    def score_industry(
        self,
        sentiment_df: pl.DataFrame,
        industry: str,
        date: str,
    ) -> dict:
        """Compute all signals for a single industry on a given date."""
        momentum = self.compute_lstm_momentum(sentiment_df, industry, date)
        heat_anomaly = self.compute_iforest_anomaly(sentiment_df, industry, date)
        composite = self.compute_lgbm_composite(sentiment_df, industry, date)

        return {
            "momentum_score": momentum,
            "heat_anomaly": heat_anomaly,
            "composite_score": composite,
            "trend_direction": 1 if momentum > 0.1 else (-1 if momentum < -0.1 else 0),
        }

    def score_all_industries(
        self,
        sentiment_df: pl.DataFrame,
        industries: list[str],
        date: str,
    ) -> pl.DataFrame:
        """Compute signals for all industries on a given date."""
        rows = []
        for industry in industries:
            signals = self.score_industry(sentiment_df, industry, date)
            rows.append({
                "industry": industry,
                "date": date,
                **signals,
            })
        return pl.DataFrame(rows)
