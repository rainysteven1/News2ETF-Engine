"""LangGraph workflow definition for the agent."""

from __future__ import annotations

from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from agent.agent.state import AgentState
from agent.agent import nodes
from agent.config import AgentRootConfig


def build_workflow(config: AgentRootConfig) -> CompiledStateGraph:
    """Build the LangGraph workflow for agent decision-making."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("read_signals", partial(nodes.read_signals, config=config))
    workflow.add_node("analyze_industry", partial(nodes.analyze_industry, config=config))
    workflow.add_node("decide_position", partial(nodes.decide_position, config=config))
    workflow.add_node("execute_trade", partial(nodes.execute_trade, config=config))

    # Set entry point
    workflow.set_entry_point("read_signals")

    # Define edges
    workflow.add_edge("read_signals", "analyze_industry")
    workflow.add_edge("analyze_industry", "decide_position")
    workflow.add_edge("decide_position", "execute_trade")
    workflow.add_edge("execute_trade", END)

    return workflow.compile()
