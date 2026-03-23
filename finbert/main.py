"""
FinBERT CLI — unified entry point for all training and inference commands.

All configuration is read from finbert/config.toml; no per-command flags needed.

Usage:
    # Train the hierarchical classifier
    python -m finbert.main train

    # Run batch inference on unlabeled news
    python -m finbert.main predict

    # Optionally point to a different config file
    python -m finbert.main --config path/to/config.toml train
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from src.config import FinBERTConfig, load_config

app = typer.Typer(
    name="finbert",
    help="FinBERT hierarchical news classifier — train and inference CLI.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

console = Console()

# Shared option: alternate config file path
_ConfigOpt = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to a TOML config file (default: finbert/config.toml)"),
]
_DEFAULT_CONFIG = Path(__file__).parent / "config.toml"


def _load(config: Path) -> FinBERTConfig:
    return load_config(config)


def _print_config_table(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title, show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(style="green")
    for key, val in rows:
        table.add_row(key, val)
    console.print(table)


@app.command()
def train(
    config: _ConfigOpt = _DEFAULT_CONFIG,
) -> None:
    """Train the FinBERT hierarchical classifier using settings from config.toml."""
    from src.train import train as _train

    cfg = _load(config)
    _print_config_table(
        "[bold]Training config[/bold]",
        [
            ("Config file", str(config)),
            ("Pretrained model", cfg.model.pretrained_model),
            ("Epochs", str(cfg.training.num_epochs)),
            ("Batch size", str(cfg.training.batch_size)),
            ("Output dir", str(cfg.output.output_dir)),
            ("W&B mode", cfg.wandb.mode),
        ],
    )
    _train(cfg)


@app.command()
def predict(
    config: _ConfigOpt = _DEFAULT_CONFIG,
) -> None:
    """Run batch inference on an unlabeled parquet file using settings from config.toml."""
    from src.predict import predict as _predict

    cfg = _load(config)
    icfg = cfg.inference
    _print_config_table(
        "[bold]Inference config[/bold]",
        [
            ("Config file", str(config)),
            ("Checkpoint", str(icfg.checkpoint_dir)),
            ("Input", str(icfg.input_file)),
            ("Output", str(icfg.output_file)),
            ("Batch size", str(icfg.infer_batch_size)),
            ("Use content", str(icfg.use_content)),
        ],
    )
    _predict(
        checkpoint_dir=icfg.checkpoint_dir,
        input_path=icfg.input_file,
        output_path=icfg.output_file,
        batch_size=icfg.infer_batch_size,
        max_length=cfg.model.max_seq_length,
        use_content=icfg.use_content,
    )


if __name__ == "__main__":
    app()
