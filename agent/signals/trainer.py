"""ML model training loop — trains LSTM, IsolationForest, and LightGBM on industry sentiment data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from loguru import logger
from sklearn.ensemble import IsolationForest as SKFIsolationForest

from agent.config import AgentRootConfig
from agent.signals.models import LSTMModel


def prepare_lstm_training_data(
    sentiment_df: pl.DataFrame,
    sequence_length: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare sequences for LSTM training.

    Returns:
        X: (n_samples, sequence_length, 1)
        y: (n_samples,) — next-day sentiment change direction
    """
    industries = sentiment_df["industry"].unique().to_list()
    all_sequences = []
    all_labels = []

    for industry in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == industry).sort("date")
        sentiment_values = ind_df["sentiment_mean"].to_numpy()

        for i in range(len(sentiment_values) - sequence_length):
            seq = sentiment_values[i : i + sequence_length]
            # Label: next-day sentiment change direction (+1, 0, -1)
            next_sent = sentiment_values[i + sequence_length]
            prev_sent = sentiment_values[i + sequence_length - 1]
            label = 1 if next_sent > prev_sent else (-1 if next_sent < prev_sent else 0)
            all_sequences.append(seq)
            all_labels.append(label)

    X = np.array(all_sequences, dtype=np.float32).reshape(-1, sequence_length, 1)
    y = np.array(all_labels, dtype=np.float32)
    return X, y


def prepare_iforest_training_data(
    sentiment_df: pl.DataFrame,
    sequence_length: int = 5,
) -> np.ndarray:
    """Prepare features for Isolation Forest training (news_count time series)."""
    industries = sentiment_df["industry"].unique().to_list()
    all_features = []

    for industry in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == industry).sort("date")
        news_counts = ind_df["news_count"].to_numpy()
        news_heat = ind_df["news_heat"].to_numpy()

        for i in range(len(news_counts) - sequence_length):
            count_seq = news_counts[i : i + sequence_length]
            heat_seq = news_heat[i : i + sequence_length]
            features = np.concatenate([count_seq, heat_seq])
            all_features.append(features)

    return np.array(all_features, dtype=np.float32)


def prepare_lgbm_training_data(
    sentiment_df: pl.DataFrame,
    sequence_length: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare features for LightGBM training.

    Features: momentum_score, heat_anomaly, sentiment_trend, news_count
    Target: next-day sentiment change direction
    """
    industries = sentiment_df["industry"].unique().to_list()
    all_features = []
    all_labels = []

    for industry in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == industry).sort("date")
        n = len(ind_df)

        sentiment_mean = ind_df["sentiment_mean"].to_numpy()
        news_count = ind_df["news_count"].to_numpy()
        news_heat = ind_df["news_heat"].to_numpy()
        sentiment_trend = ind_df["sentiment_trend"].to_numpy()

        for i in range(sequence_length, n - 1):
            # Features from the lookback window
            mom_score = sentiment_mean[i] - sentiment_mean[i - sequence_length]
            heat_val = news_heat[i]
            trend_val = sentiment_trend[i] if i < len(sentiment_trend) else 0.0
            count_val = news_count[i]

            features = np.array([mom_score, heat_val, trend_val, count_val], dtype=np.float32)
            all_features.append(features)

            # Label: next day direction
            next_sent = sentiment_mean[i + 1]
            curr_sent = sentiment_mean[i]
            label = 1 if next_sent > curr_sent else (-1 if next_sent < curr_sent else 0)
            all_labels.append(label)

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def train_lstm(
    sentiment_df: pl.DataFrame,
    config: AgentRootConfig,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
) -> Path:
    """Train LSTM model and save checkpoint."""
    logger.info("Preparing LSTM training data...")
    X, y = prepare_lstm_training_data(
        sentiment_df, sequence_length=config.model.lstm.sequence_length
    )
    logger.info("LSTM training set: {} samples", len(X))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(
        hidden_size=config.model.lstm.hidden_size,
        num_layers=config.model.lstm.num_layers,
        dropout=config.model.lstm.dropout,
    ).to(device)

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info("LSTM Epoch {}/{} — loss: {:.4f}", epoch + 1, epochs, total_loss / len(loader))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lstm_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("LSTM model saved to {}", model_path)
    return model_path


def train_isolation_forest(
    sentiment_df: pl.DataFrame,
    config: AgentRootConfig,
    output_dir: Path,
) -> Path:
    """Train Isolation Forest model and save checkpoint."""
    logger.info("Preparing Isolation Forest training data...")
    X = prepare_iforest_training_data(
        sentiment_df, sequence_length=config.model.lstm.sequence_length
    )
    logger.info("Isolation Forest training set: {} samples", len(X))

    model = SKFIsolationForest(
        contamination=config.model.isolation_forest.contamination,
        n_estimators=config.model.isolation_forest.n_estimators,
        random_state=42,
    )
    model.fit(X)

    output_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    model_path = output_dir / "iforest_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Isolation Forest model saved to {}", model_path)
    return model_path


def train_lightgbm(
    sentiment_df: pl.DataFrame,
    config: AgentRootConfig,
    output_dir: Path,
) -> Path:
    """Train LightGBM model and save checkpoint."""
    logger.info("Preparing LightGBM training data...")
    X, y = prepare_lgbm_training_data(
        sentiment_df, sequence_length=config.model.lstm.sequence_length
    )
    logger.info("LightGBM training set: {} samples, features: {}", len(X), X.shape[1])

    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm is required for LightGBM training") from e

    model = lgb.LGBMRegressor(
        num_leaves=config.model.lightgbm.num_leaves,
        learning_rate=config.model.lightgbm.learning_rate,
        n_estimators=config.model.lightgbm.n_estimators,
        verbose=-1,
    )
    model.fit(X, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lgbm_model.txt"
    model.booster_.save_model(str(model_path))
    logger.info("LightGBM model saved to {}", model_path)
    return model_path


def train_all_models(
    sentiment_df: pl.DataFrame,
    config: AgentRootConfig,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Train all ML models and return paths to saved checkpoints."""
    if output_dir is None:
        output_dir = Path("agent/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting model training...")
    results = {}

    logger.info("=" * 60)
    logger.info("Training LSTM...")
    results["lstm"] = train_lstm(sentiment_df, config, output_dir)

    logger.info("=" * 60)
    logger.info("Training Isolation Forest...")
    results["iforest"] = train_isolation_forest(sentiment_df, config, output_dir)

    logger.info("=" * 60)
    logger.info("Training LightGBM...")
    results["lgbm"] = train_lightgbm(sentiment_df, config, output_dir)

    logger.info("=" * 60)
    logger.info("All models trained successfully!")
    return results
