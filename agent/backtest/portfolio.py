"""Portfolio management for backtesting."""

from __future__ import annotations


import polars as pl


class Portfolio:
    """Simple portfolio manager for tracking holdings and NAV."""

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        transaction_fee: float = 0.0003,
        slippage: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings: dict[str, float] = {}  # industry -> weight
        self.transaction_fee = transaction_fee
        self.slippage = slippage

    @property
    def nav(self) -> float:
        """Current net asset value (cash + holdings value)."""
        return self.cash

    @property
    def total_weight(self) -> float:
        return sum(self.holdings.values())

    def update_holdings(
        self,
        decisions: list[dict],
        etf_prices: pl.DataFrame,
        date: str,
    ) -> None:
        """Update holdings based on trade decisions.

        Args:
            decisions: List of TradeDecision dicts
            etf_prices: DataFrame with columns: date, etf, close
            date: Current date string
        """
        # Calculate target holdings from decisions
        target_holdings: dict[str, float] = {}
        for decision in decisions:
            industry = decision["industry"]
            action = decision["action"]
            weight = decision["weight"]

            if action == "buy":
                target_holdings[industry] = weight
            elif action == "sell":
                target_holdings[industry] = 0.0
            elif action == "hold":
                # Keep existing or increase to new weight
                current = self.holdings.get(industry, 0.0)
                target_holdings[industry] = max(current, weight)

        # Apply transaction costs and slippage when changing positions
        for industry, target_weight in target_holdings.items():
            current_weight = self.holdings.get(industry, 0.0)
            if abs(target_weight - current_weight) > 0.001:
                # Position change — apply costs
                cost = abs(target_weight - current_weight) * (self.transaction_fee + self.slippage)
                self.cash -= cost * self.nav

        self.holdings = {k: v for k, v in target_holdings.items() if v > 0.001}

    def compute_daily_return(
        self,
        etf_prices: pl.DataFrame,
        date: str,
        industry_etf_map: dict[str, list[str]],
    ) -> float:
        """Compute daily portfolio return based on ETF price changes.

        Args:
            etf_prices: DataFrame with date, etf, close columns
            date: Current date
            industry_etf_map: industry -> list of ETF names

        Returns:
            Daily return as a float (e.g., 0.01 = 1%)
        """
        date_df = etf_prices.filter(pl.col("date") == date)
        if len(date_df) == 0 or self.total_weight == 0:
            return 0.0

        # For simplicity, use equal weight across ETFs in each industry
        industry_returns = []
        for industry, weight in self.holdings.items():
            etfs = industry_etf_map.get(industry, [])
            if not etfs:
                continue

            # Get average return for this industry's ETFs
            etf_returns = []
            for etf in etfs:
                etf_df = date_df.filter(pl.col("etf") == etf)
                if len(etf_df) >= 2:
                    prev_close = etf_df["close"].to_list()[0]
                    curr_close = etf_df["close"].to_list()[-1]
                    ret = (curr_close - prev_close) / prev_close
                    etf_returns.append(ret)

            if etf_returns:
                industry_returns.append(weight * sum(etf_returns) / len(etf_returns))

        if not industry_returns:
            return 0.0
        return sum(industry_returns)

    def update_nav(self, daily_return: float) -> None:
        """Update NAV based on daily return."""
        self.cash *= 1 + daily_return

    def record_state(
        self,
        date: str,
        strategy_id: str = "default",
    ) -> dict:
        """Record current portfolio state for backtest results."""
        return {
            "strategy_id": strategy_id,
            "date": date,
            "holdings": self.holdings.copy(),
            "cash": self.cash,
            "nav": self.nav,
            "daily_return": 0.0,  # Will be filled by engine
            "cumulative_return": (self.nav - self.initial_capital) / self.initial_capital,
        }
