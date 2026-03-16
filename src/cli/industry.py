import typer
from rich.console import Console

from src.industry import DEFAULT_MODEL, build_industry_dict

app = typer.Typer(help="Industry related commands")
console = Console()


@app.command(name="build-dict")
def build_dict(
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", "-m", help="ZhipuAI 模型名称，如 glm-4-flash / glm-4-plus / glm-4"
    ),
):
    """
    Build hierarchical ETF industry dictionary via ZhipuAI and save to data/industry_dict.json.

    Requires ZHIPU_API_KEY environment variable.

    Example:
        python main.py build-dict
        python main.py build-dict --model glm-4-plus
    """
    import os

    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        console.print("[bold red]请设置环境变量 ZHIPU_API_KEY[/bold red]")
        raise typer.Exit(1)

    build_industry_dict(api_key=api_key, console=console, model=model)
