"""Walk-forward backtesting engine."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from loguru import logger

from agent.agent.state import AgentState
from agent.agent.workflow import build_workflow
from agent.backtest.metrics import calculate_metrics
from agent.backtest.portfolio import Portfolio
from agent.config import AgentRootConfig
from agent.utils.industry_map import IndustryMapper


class WalkForwardEngine:
    """Walk-forward backtesting engine that runs the agent workflow day by day."""

    def __init__(
        self,
        config: AgentRootConfig,
        checkpoint_dir: Path | None = None,
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir or Path("agent/checkpoints")

        # Initialize industry mapper
        self.mapper = IndustryMapper(config.data.industry_dict)

        # Build agent workflow
        self.workflow = build_workflow(config)

        # ETF price data (loaded lazily)
        self._etf_prices: pl.DataFrame | None = None

        # Industry to ETF mapping for NAV calculation
        self._industry_etf_map: dict[str, list[str]] = {}
        for industry in self.mapper.industries:
            self._industry_etf_map[industry] = self.mapper.industry_etfs(industry)

    def _load_etf_prices(self) -> pl.DataFrame | None:
        """Load ETF prices lazily."""
        if self._etf_prices is None:
            path = self.config.data.etf_prices
            if path.exists():
                self._etf_prices = pl.read_parquet(path)
                logger.info("Loaded ETF prices: {} rows", len(self._etf_prices))
            else:
                logger.warning("ETF prices file not found at {}", path)
        return self._etf_prices

    def _get_trading_days(
        self,
        start_date: str,
        end_date: str,
    ) -> list[str]:
        """Get list of trading days between start and end date."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        trading_days = []
        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:
                trading_days.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return trading_days

    def run(
        self,
        start_date: str,
        end_date: str,
        run_id: str | None = None,
    ) -> pl.DataFrame:
        """Run the backtest over the specified date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            run_id: Optional run identifier for tracking

        Returns:
            DataFrame with backtest results
        """
        if run_id is None:
            run_id = f"bt_{uuid.uuid4().hex[:8]}"

        logger.info("Starting backtest from {} to {}, run_id={}", start_date, end_date, run_id)

        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.config.backtest.initial_capital,
            transaction_fee=self.config.backtest.transaction_fee,
            slippage=self.config.backtest.slippage,
        )

        # Get trading days
        trading_days = self._get_trading_days(start_date, end_date)
        logger.info("Total trading days: {}", len(trading_days))

        # Load ETF prices for NAV calculation
        etf_prices = self._load_etf_prices()

        # Load ML signals
        signals_path = self.config.data.output_signals
        if not signals_path.exists():
            logger.error("ML signals not found at {}. Run compute-signals first.", signals_path)
            return pl.DataFrame()

        signals_df = pl.read_parquet(signals_path)
        logger.info("Loaded ML signals: {} rows", len(signals_df))

        # Get all unique dates from signals that fall within range
        available_dates = (
            signals_df.filter(pl.col("date") >= start_date)
            .filter(pl.col("date") <= end_date)
            .select("date")
            .unique()
            .sort("date")["date"]
            .to_list()
        )

        results = []

        for i, date in enumerate(available_dates):
            date_str = str(date)
            logger.debug("Processing date: {} ({}/{})", date_str, i + 1, len(available_dates))

            # Get signals for this date
            day_signals = signals_df.filter(pl.col("date") == date_str)
            signals_dict: dict[str, dict] = {}
            for row in day_signals.iter_rows(named=True):
                signals_dict[row["industry"]] = {
                    "momentum_score": row["momentum_score"],
                    "heat_anomaly": row["heat_anomaly"],
                    "composite_score": row["composite_score"],
                    "trend_direction": row["trend_direction"],
                }

            # Build initial state
            state: AgentState = AgentState(
                date=date_str,
                run_id=run_id,
                signals=signals_dict,
                holdings=portfolio.holdings,
                cash=portfolio.cash,
                decisions=[],
                reasoning="",
                error=None,
            )

            # Run agent workflow
            try:
                result_state = self.workflow.invoke(state)
                portfolio.holdings = dict(result_state.get("holdings", {}))
                portfolio.cash = result_state.get("cash", portfolio.cash)
                decisions = result_state.get("decisions", [])
            except Exception as e:
                logger.error("Workflow failed for {}: {}", date_str, e)
                decisions = []

            # Compute daily return using ETF prices
            daily_return = 0.0
            if etf_prices is not None and portfolio.total_weight > 0:
                daily_return = portfolio.compute_daily_return(
                    etf_prices,
                    date_str,
                    self._industry_etf_map,
                )

            # Update portfolio NAV
            portfolio.update_nav(daily_return)

            # Record results
            result = portfolio.record_state(date_str, run_id)
            result["daily_return"] = daily_return
            result["cumulative_return"] = (portfolio.nav - portfolio.initial_capital) / portfolio.initial_capital
            results.append(result)

        # Save backtest results
        results_df = pl.DataFrame(results)
        output_path = self.config.data.output_backtest
        results_df.write_parquet(output_path)
        logger.info("Backtest complete. Results saved to {}", output_path)

        # Calculate and log metrics
        metrics = calculate_metrics(
            results_df,
            risk_free_rate=self.config.backtest.risk_free_rate,
        )
        logger.info("=" * 60)
        logger.info("Backtest Results for run_id={}", run_id)
        for k, v in metrics.items():
            logger.info("  {}: {}", k, v)
        logger.info("=" * 60)

        return results_df
