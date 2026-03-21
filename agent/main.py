"""Agent module CLI — unified entry point for ML signals, agent decisions, and backtesting.

Usage:
    # Train ML models
    python -m agent.main train-signals

    # Compute ML signals for a date range (batch pre-computation)
    python -m agent.main compute-signals --start-date 2021-01-01 --end-date 2024-12-31

    # Run backtest
    python -m agent.main backtest --start-date 2021-01-01 --end-date 2024-12-31

    # Single-day agent decision (debug mode)
    python -m agent.main decide --date 2023-06-15
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from agent.agent.state import AgentState
from agent.agent.workflow import build_workflow
from agent.backtest.engine import WalkForwardEngine
from agent.config import AgentRootConfig, load_config
from agent.signals import trainer
from agent.signals.features import aggregate_industry_sentiment
from agent.signals.scorer import SignalScorer

app = typer.Typer(
    name="agent",
    help="Agent module — ML signal computation, LangGraph agent decisions, and backtesting.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

console = Console()

_DEFAULT_CONFIG = Path(__file__).parent / "config.toml"


def _load(config: Path) -> AgentRootConfig:
    return load_config(config)


def _print_config_table(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title, show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(style="green")
    for key, val in rows:
        table.add_row(key, val)
    console.print(table)


@app.command()
def train_signals(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to config file"),
    ]
    | None = None,
) -> None:
    """Train LSTM, IsolationForest, and LightGBM models on industry sentiment data."""
    config_path = config or _DEFAULT_CONFIG
    cfg = _load(config_path)

    _print_config_table(
        "[bold]Train Signals Config[/bold]",
        [
            ("Config file", str(config_path)),
            ("LSTM hidden_size", str(cfg.model.lstm.hidden_size)),
            ("LSTM num_layers", str(cfg.model.lstm.num_layers)),
            ("IForest contamination", str(cfg.model.isolation_forest.contamination)),
            ("LightGBM n_estimators", str(cfg.model.lightgbm.n_estimators)),
        ],
    )

    # Load or compute industry sentiment
    sentiment_path = cfg.data.output_sentiment
    if sentiment_path.exists():
        logger.info("Loading existing industry sentiment from {}", sentiment_path)
        sentiment_df = pl.read_parquet(sentiment_path)
    else:
        logger.info("Computing industry sentiment from classified news...")
        sentiment_df = aggregate_industry_sentiment(
            classified_path=cfg.data.input_classified,
            industry_dict_path=cfg.data.industry_dict,
            output_path=sentiment_path,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )

    # Train models
    output_dir = Path("agent/checkpoints")
    results = trainer.train_all_models(sentiment_df, cfg, output_dir)

    console.print("[bold green]Models trained successfully![/bold green]")
    for name, path in results.items():
        console.print(f"  {name}: {path}")


@app.command()
def compute_signals(
    start_date: Annotated[
        str,
        typer.Option("--start-date", help="Start date (YYYY-MM-DD)"),
    ],
    end_date: Annotated[
        str,
        typer.Option("--end-date", help="End date (YYYY-MM-DD)"),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to config file"),
    ]
    | None = None,
) -> None:
    """Batch pre-compute ML signals for a date range and save to ml_signals.parquet."""
    config_path = config or _DEFAULT_CONFIG
    cfg = _load(config_path)

    _print_config_table(
        "[bold]Compute Signals Config[/bold]",
        [
            ("Config file", str(config_path)),
            ("Start date", start_date),
            ("End date", end_date),
            ("Output", str(cfg.data.output_signals)),
        ],
    )

    # Load or compute industry sentiment
    sentiment_path = cfg.data.output_sentiment
    if sentiment_path.exists():
        logger.info("Loading industry sentiment from {}", sentiment_path)
        sentiment_df = pl.read_parquet(sentiment_path)
    else:
        logger.info("Computing industry sentiment first...")
        sentiment_df = aggregate_industry_sentiment(
            classified_path=cfg.data.input_classified,
            industry_dict_path=cfg.data.industry_dict,
            output_path=sentiment_path,
            start_date=start_date,
            end_date=end_date,
        )

    # Initialize scorer
    checkpoint_dir = Path("agent/checkpoints")
    if not checkpoint_dir.exists():
        console.print("[bold red]Error: No checkpoints found. Run train-signals first.[/bold red]")
        raise typer.Exit(1)

    scorer = SignalScorer(cfg, checkpoint_dir)

    # Get trading days
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    trading_days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            trading_days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    logger.info("Computing signals for {} trading days...", len(trading_days))

    # Get industries
    industries = sentiment_df["industry"].unique().to_list()

    all_signals = []
    for i, date in enumerate(trading_days):
        if i % 50 == 0:
            logger.info("Progress: {}/{}", i, len(trading_days))

        day_signals = scorer.score_all_industries(sentiment_df, industries, date)
        all_signals.append(day_signals)

    if all_signals:
        signals_df = pl.concat(all_signals)
        output_path = cfg.data.output_signals
        signals_df.write_parquet(output_path)
        logger.info("Saved {} signal rows to {}", len(signals_df), output_path)
        console.print(f"[bold green]Signals computed and saved to {output_path}[/bold green]")
    else:
        console.print("[bold yellow]No signals computed (check data availability)[/bold yellow]")


@app.command()
def backtest(
    start_date: Annotated[
        str,
        typer.Option("--start-date", help="Start date (YYYY-MM-DD)"),
    ],
    end_date: Annotated[
        str,
        typer.Option("--end-date", help="End date (YYYY-MM-DD)"),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to config file"),
    ]
    | None = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Run identifier"),
    ] = None,
) -> None:
    """Run walk-forward backtest using pre-computed ML signals and LangGraph agent."""
    config_path = config or _DEFAULT_CONFIG
    cfg = _load(config_path)

    if run_id is None:
        run_id = f"bt_{uuid.uuid4().hex[:8]}"

    _print_config_table(
        "[bold]Backtest Config[/bold]",
        [
            ("Config file", str(config_path)),
            ("Start date", start_date),
            ("End date", end_date),
            ("Run ID", run_id),
            ("Initial capital", f"{cfg.backtest.initial_capital:,.2f}"),
        ],
    )

    engine = WalkForwardEngine(cfg, checkpoint_dir=Path("agent/checkpoints"))
    results_df = engine.run(start_date, end_date, run_id=run_id)

    if len(results_df) > 0:
        console.print(f"[bold green]Backtest complete! Results: {len(results_df)} trading days[/bold green]")
    else:
        console.print("[bold red]Backtest produced no results. Check signal files.[/bold red]")


@app.command()
def decide(
    date: Annotated[
        str,
        typer.Option("--date", help="Date to make decision (YYYY-MM-DD)"),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to config file"),
    ]
    | None = None,
) -> None:
    """Run single-day agent decision (debug mode, does not write to portfolio)."""
    config_path = config or _DEFAULT_CONFIG
    cfg = _load(config_path)

    _print_config_table(
        "[bold]Decide Config[/bold]",
        [
            ("Config file", str(config_path)),
            ("Date", date),
            ("LLM model", cfg.agent.llm_model),
        ],
    )

    # Load signals
    signals_path = cfg.data.output_signals
    if not signals_path.exists():
        console.print(f"[bold red]Error: Signals file not found at {signals_path}[/bold red]")
        console.print("Run compute-signals first to generate ML signals.")
        raise typer.Exit(1)

    signals_df = pl.read_parquet(signals_path)
    day_signals = signals_df.filter(pl.col("date") == date)

    if len(day_signals) == 0:
        console.print(f"[bold yellow]No signals found for date {date}[/bold yellow]")
        raise typer.Exit(0)

    # Build signals dict
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
        date=date,
        run_id=f"debug_{uuid.uuid4().hex[:8]}",
        signals=signals_dict,
        holdings={},
        cash=cfg.backtest.initial_capital,
        decisions=[],
        reasoning="",
        error=None,
    )

    # Run agent workflow
    workflow = build_workflow(cfg)
    console.print(f"[bold cyan]Running agent workflow for {date}...[/bold cyan]")

    try:
        result_state = workflow.invoke(state)

        console.print("\n[bold]=== Agent Reasoning ===[/bold]")
        console.print(result_state.get("reasoning", "No reasoning generated."))

        decisions = result_state.get("decisions", [])
        console.print(f"\n[bold]=== Trade Decisions ({len(decisions)}) ===[/bold]")
        for d in decisions:
            console.print(f"  {d['industry']}: {d['action']} {d['weight']:.3f} — {d['reason']}")

        if result_state.get("error"):
            console.print(f"\n[bold yellow]Errors:[/bold yellow] {result_state['error']}")

    except Exception as e:
        console.print(f"[bold red]Workflow failed:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
