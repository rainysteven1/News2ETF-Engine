"""LangGraph node functions for the agent workflow."""

from __future__ import annotations

import json

import polars as pl
from loguru import logger

from agent.agent.client import LLMClient
from agent.agent.prompts import get_prompt
from agent.agent.state import AgentState, TradeDecision
from agent.config import AgentRootConfig


def read_signals(state: AgentState, config: AgentRootConfig) -> AgentState:
    """Read ML signals for the given date from ml_signals.parquet."""
    date = state.date
    signals_path = config.data.output_signals

    logger.info("Reading signals for date {}", date)

    if not signals_path.exists():
        logger.warning("Signals file not found at {}, using empty signals", signals_path)
        state.signals = {}
        return state

    df = pl.read_parquet(signals_path)
    df = df.filter(pl.col("date") == date)

    signals: dict[str, dict] = {}
    for row in df.iter_rows(named=True):
        industry = row["industry"]
        signals[industry] = {
            "momentum_score": row["momentum_score"],
            "heat_anomaly": row["heat_anomaly"],
            "composite_score": row["composite_score"],
            "trend_direction": row["trend_direction"],
        }

    state.signals = signals
    logger.info("Loaded signals for {} industries", len(signals))
    return state


def analyze_industry(state: AgentState, config: AgentRootConfig) -> AgentState:
    """Use LLM to analyze industry signals and generate market overview."""
    client = LLMClient(
        model=config.agent.llm_model,
        temperature=config.agent.llm_temperature,
    )

    # Build signals summary
    signals_summary_lines = []
    for industry, sigs in state.signals.items():
        signals_summary_lines.append(
            f"- {industry}: momentum={sigs['momentum_score']:.3f}, "
            f"heat={sigs['heat_anomaly']:.3f}, composite={sigs['composite_score']:.3f}"
        )
    signals_summary = "\n".join(signals_summary_lines) if signals_summary_lines else "No signals available."

    system_prompt = "You are a financial analyst specializing in sector rotation based on news sentiment."
    user_prompt = get_prompt(
        "analyze_industry",
        date=state.date,
        signals_summary=signals_summary,
    )

    try:
        reasoning = client.chat(system_prompt, user_prompt)
        state.reasoning = reasoning
        logger.debug("Analysis: {}", reasoning[:200])
    except Exception as e:
        logger.error("LLM analysis failed: {}", e)
        state.error = str(e)
        state.reasoning = "Analysis unavailable due to error."

    return state


def decide_position(state: AgentState, config: AgentRootConfig) -> AgentState:
    """Use LLM to decide positions based on signals and holdings."""
    client = LLMClient(
        model=config.agent.llm_model,
        temperature=config.agent.llm_temperature,
    )

    # Build holdings summary
    holdings_lines = []
    for industry, weight in state.holdings.items():
        if weight > 0:
            holdings_lines.append(f"- {industry}: {weight:.3f}")
    holdings_summary = "\n".join(holdings_lines) if holdings_lines else "No current holdings."

    system_prompt = "You are a quantitative portfolio manager making sector allocation decisions."
    user_prompt = get_prompt(
        "decide_position",
        analysis=state.reasoning or "No analysis available.",
        holdings=holdings_summary,
        max_weight_per_industry=config.agent.max_weight_per_industry,
        max_total_weight=config.agent.max_total_weight,
        cash=state.cash or 0.0,
    )

    try:
        decisions_data = client.chat_json(system_prompt, user_prompt)
        decisions: list[TradeDecision] = []
        for item in decisions_data:
            decisions.append(
                TradeDecision(
                    industry=item["industry"],
                    action=item.get("action", "hold"),
                    weight=float(item.get("weight", 0.0)),
                    etf=item.get("etf"),
                    reason=item.get("reason", ""),
                )
            )
        state.decisions = decisions
        logger.info("Generated {} trade decisions", len(decisions))
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM JSON response: {}", e)
        state.decisions = []
        state.error = f"Failed to parse decisions: {e}"
    except Exception as e:
        logger.error("LLM decision failed: {}", e)
        state.decisions = []
        state.error = str(e)

    return state


def execute_trade(state: AgentState, config: AgentRootConfig) -> AgentState:
    """Execute trades by updating holdings and writing trade signals to file."""
    from agent.backtest.portfolio import Portfolio

    decisions = state.decisions
    if not decisions:
        logger.info("No trades to execute")
        return state

    # Load or create portfolio
    holdings = dict(state.holdings)
    cash = state.cash or config.backtest.initial_capital

    # Apply decisions
    for decision in decisions:
        industry = decision.industry
        action = decision.action
        weight = decision.weight

        if action == "buy":
            holdings[industry] = weight
        elif action == "sell":
            holdings[industry] = 0.0
        elif action == "hold":
            if industry in holdings:
                holdings[industry] = max(holdings[industry], weight)
        else:
            logger.warning("Unknown action '{}' for {}", action, industry)

    # Normalize holdings
    total_weight = sum(holdings.values())
    if total_weight > config.agent.max_total_weight:
        for industry in holdings:
            holdings[industry] = holdings[industry] / total_weight * config.agent.max_total_weight

    state.holdings = holdings

    # Write trade signals
    output_path = config.data.output_trades
    trade_rows = [
        {
            "date": state.date,
            "run_id": state.run_id,
            **d.model_dump(),
        }
        for d in decisions
    ]
    trade_df = pl.DataFrame(trade_rows)

    if output_path.exists():
        existing = pl.read_parquet(output_path)
        trade_df = pl.concat([existing, trade_df])
    trade_df.write_parquet(output_path)
    logger.info("Wrote {} trade signals to {}", len(trade_df), output_path)

    # Write decision logs
    log_path = config.data.output_logs
    log_entry = {
        "date": state.date,
        "run_id": state.run_id,
        "input_signals": state.signals,
        "reasoning_text": state.reasoning or "",
        "output_decision": decisions,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return state
