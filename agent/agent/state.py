"""Agent state definition for LangGraph workflow."""

from __future__ import annotations

from pydantic import BaseModel


class TradeDecision(BaseModel):
    """A single trading decision for an industry."""

    industry: str
    action: str  # "buy", "sell", "hold"
    weight: float
    etf: str | None = None
    reason: str = ""


class AgentState(BaseModel):
    """State passed through the LangGraph workflow."""

    date: str = ""
    run_id: str = ""
    signals: dict[str, dict] = {}  # industry -> MLSignal
    holdings: dict[str, float] = {}  # industry -> weight
    cash: float = 0.0
    decisions: list[TradeDecision] = []
    reasoning: str = ""
    error: str | None = None
