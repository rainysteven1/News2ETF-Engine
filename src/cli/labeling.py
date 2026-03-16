import typer
from pydantic import BaseModel
from rich.console import Console

from src.common.tracking import track
from src.labeling import print_label_stats, run_level1, run_level2
from src.store import DB_PATH, ensure_news_loaded, init_db

app = typer.Typer(help="Labeling related commands")
console = Console()


class LabelParams(BaseModel):
    level: int
    sample: int
    force: bool = False


class LabelResult(BaseModel):
    total_labeled: int = 0
    keyword_count: int = 0
    llm_count: int = 0


@track("label_runs", LabelParams, LabelResult)
def _run_label(params: LabelParams, *, run_id: str | None = None) -> LabelResult:
    import os

    import duckdb

    try:
        con = duckdb.connect(str(DB_PATH))
    except duckdb.IOException as e:
        console.print(
            f"[bold red]Cannot open database:[/bold red] {e}\n"
            "[yellow]Another process is holding the lock. Close any open DuckDB sessions and retry.[/yellow]"
        )
        raise typer.Exit(1)
    init_db(con)
    ensure_news_loaded(con, console)

    # Check if there is anything left to label (skip expensive API calls when nothing to do)
    if params.level == 1:
        pending = con.execute("""
            SELECT COUNT(*) FROM news_raw r
            LEFT JOIN news_classified c ON r.news_id = c.news_id
            WHERE c.news_id IS NULL AND r.title IS NOT NULL AND LENGTH(r.title) > 5
        """).fetchone()[0]  # type: ignore
        if pending == 0 and not params.force:
            already = con.execute("SELECT COUNT(*) FROM news_classified").fetchone()[0]  # type: ignore
            console.print(
                f"[yellow]All news already labeled at Level-1 ({already} records). Skipping. "
                f"Use --force to wipe and re-label.[/yellow]"
            )
            con.close()
            return LabelResult(total_labeled=already)
        if params.force:
            con.execute("DELETE FROM news_classified")
            console.print("[yellow]--force: cleared news_classified, re-labeling from scratch[/yellow]")
    else:
        pending = con.execute("""
            SELECT COUNT(*) FROM news_classified
            WHERE major_category IS NOT NULL AND sub_category IS NULL
        """).fetchone()[0]  # type: ignore
        if pending == 0 and not params.force:
            already = con.execute("SELECT COUNT(*) FROM news_classified WHERE sub_category IS NOT NULL").fetchone()[0]  # type: ignore
            console.print(
                f"[yellow]所有 Level-1 记录已完成 Level-2 标注（共 {already} 条），跳过。"
                f"使用 --force 强制重标。[/yellow]"
            )
            con.close()
            return LabelResult(total_labeled=already)
        if params.force:
            con.execute("""
                UPDATE news_classified
                SET sub_category = NULL, sentiment = NULL
                WHERE sub_category IS NOT NULL
            """)
            console.print("[yellow]--force: reset all sub_category/sentiment, re-labeling[/yellow]")

    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        console.print("[bold red]Environment variable ZHIPU_API_KEY is not set.[/bold red]")
        con.close()
        raise typer.Exit(1)

    from zhipuai import ZhipuAI

    client = ZhipuAI(api_key=api_key)

    function = run_level1 if params.level == 1 else run_level2
    function(con, client, params.sample, run_id=run_id)

    total = con.execute("SELECT COUNT(*) FROM news_classified").fetchone()[0]  # type: ignore
    kw = con.execute("SELECT COUNT(*) FROM news_classified WHERE label_source = 'keyword'").fetchone()[0]  # type: ignore

    print_label_stats(con, console)
    con.close()

    return LabelResult(total_labeled=total, keyword_count=kw, llm_count=total - kw)


@app.command()
def show_stats():
    """
    Show labeling statistics without needing API access.

    Example:
        python main.py show-stats
    """
    import duckdb

    try:
        con = duckdb.connect(str(DB_PATH))
    except duckdb.IOException as e:
        console.print(
            f"[bold red]Cannot open database:[/bold red] {e}\n"
            "[yellow]Another process is holding the lock. Close any open DuckDB sessions and retry.[/yellow]"
        )
        raise typer.Exit(1)
    init_db(con)
    ensure_news_loaded(con, console)
    print_label_stats(con, console)
    con.close()


@app.command()
def label(
    sample: int = typer.Option(10000, "--sample", "-n", help="Number of news to sample"),
    level: int = typer.Option(
        1, "--level", "-l", min=1, max=2, help="Classification level: 1=major category, 2=sub category+sentiment"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-label: clear existing labels and rerun"),
):
    """
    Label news with hierarchical industry categories using keyword pre-classification + LLM fallback.

    Run Level-1 first, then Level-2.
    Requires ZHIPU_API_KEY environment variable.

    Example:
        python main.py label --level 1 --sample 10000
        python main.py label --level 2 --sample 50000
        python main.py label --level 1 --force
    """
    _run_label(LabelParams(level=level, sample=sample, force=force))
