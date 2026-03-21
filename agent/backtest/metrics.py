"""Performance metrics calculation for backtest results."""

from __future__ import annotations

import polars as pl
import numpy as np


def calculate_metrics(
    backtest_df: pl.DataFrame,
    risk_free_rate: float = 0.03,
) -> dict:
    """Calculate performance metrics from backtest results.

    Args:
        backtest_df: DataFrame with columns: date, nav, daily_return
        risk_free_rate: Annual risk-free rate

    Returns:
        Dict with key metrics: total_return, annual_return, max_drawdown, sharpe_ratio, etc.
    """
    if len(backtest_df) == 0:
        return {}

    nav_series = backtest_df["nav"].to_numpy()
    daily_returns = backtest_df["daily_return"].to_numpy()

    # Total return
    total_return = (nav_series[-1] - nav_series[0]) / nav_series[0]

    # Annualized return (assuming 252 trading days)
    n_days = len(nav_series)
    n_years = n_days / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Cumulative max drawdown
    cummax = np.maximum.accumulate(nav_series)
    drawdowns = (nav_series - cummax) / cummax
    max_drawdown = drawdowns.min()  # Most negative value

    # Sharpe ratio
    if daily_returns.std() > 0:
        daily_rf = risk_free_rate / 252
        excess_returns = daily_returns - daily_rf
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    else:
        sharpe_ratio = 0.0

    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Win rate
    positive_days = (daily_returns > 0).sum()
    win_rate = positive_days / n_days if n_days > 0 else 0.0

    # Sortino ratio
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        daily_rf = risk_free_rate / 252
        excess_returns = daily_returns - daily_rf
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    else:
        sortino_ratio = 0.0

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "calmar_ratio": float(calmar_ratio),
        "win_rate": float(win_rate),
        "trading_days": int(n_days),
        "final_nav": float(nav_series[-1]),
        "initial_capital": float(nav_series[0]),
    }
