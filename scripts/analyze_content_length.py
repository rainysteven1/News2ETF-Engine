"""
Analyze title and content lengths in news_raw.

Outputs:
  - Per-field stats: max, mean, p50, p90, p95, p99
  - Total (title + content) length stats
  - Distribution histogram in the terminal

Usage:
  uv run scripts/analyze_content_length.py
"""

from pathlib import Path

import duckdb
from rich.console import Console
from rich.table import Table

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "news2etf.duckdb"

console = Console()


def main() -> None:
    if not DB_PATH.exists():
        console.print(f"[red]Database not found: {DB_PATH}[/red]")
        return

    con = duckdb.connect(str(DB_PATH), read_only=True)

    console.print("\n[bold cyan]Fetching length stats from news_raw...[/bold cyan]")

    stats = con.execute("""
        SELECT
            COUNT(*)                                        AS total_rows,

            -- title
            MAX(LENGTH(title))                              AS title_max,
            ROUND(AVG(LENGTH(title)), 1)                   AS title_avg,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY LENGTH(title))  AS title_p50,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY LENGTH(title))  AS title_p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LENGTH(title))  AS title_p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LENGTH(title))  AS title_p99,

            -- content
            MAX(LENGTH(content))                            AS content_max,
            ROUND(AVG(LENGTH(content)), 1)                 AS content_avg,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY LENGTH(content)) AS content_p50,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY LENGTH(content)) AS content_p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LENGTH(content)) AS content_p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LENGTH(content)) AS content_p99,

            -- total (title + content)
            MAX(LENGTH(title) + LENGTH(content))                              AS total_max,
            ROUND(AVG(LENGTH(title) + LENGTH(content)), 1)                   AS total_avg,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY LENGTH(title) + LENGTH(content)) AS total_p50,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY LENGTH(title) + LENGTH(content)) AS total_p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LENGTH(title) + LENGTH(content)) AS total_p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LENGTH(title) + LENGTH(content)) AS total_p99
        FROM news_raw
        WHERE title IS NOT NULL
    """).fetchone()

    if stats is None:
        console.print("[red]No data found in news_raw.[/red]")
        return

    (
        total_rows,
        title_max,
        title_avg,
        title_p50,
        title_p90,
        title_p95,
        title_p99,
        content_max,
        content_avg,
        content_p50,
        content_p90,
        content_p95,
        content_p99,
        ttl_max,
        ttl_avg,
        ttl_p50,
        ttl_p90,
        ttl_p95,
        ttl_p99,
    ) = stats

    console.print(f"  Total rows: [bold]{total_rows:,}[/bold]\n")

    table = Table(title="Length Statistics (chars)", show_lines=True)
    table.add_column("Field", style="cyan", min_width=18)
    table.add_column("max", justify="right")
    table.add_column("avg", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p90", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("p99", justify="right")

    def fmt(v: float | int | None) -> str:
        if v is None:
            return "-"
        return f"{int(v):,}"

    table.add_row(
        "title", fmt(title_max), fmt(title_avg), fmt(title_p50), fmt(title_p90), fmt(title_p95), fmt(title_p99)
    )
    table.add_row(
        "content",
        fmt(content_max),
        fmt(content_avg),
        fmt(content_p50),
        fmt(content_p90),
        fmt(content_p95),
        fmt(content_p99),
    )
    table.add_row(
        "[bold]total[/bold]", fmt(ttl_max), fmt(ttl_avg), fmt(ttl_p50), fmt(ttl_p90), fmt(ttl_p95), fmt(ttl_p99)
    )
    console.print(table)

    # Distribution: how many rows fall within common content truncation thresholds
    console.print("\n[bold]Content length distribution (truncation impact)[/bold]")
    thresholds = [300, 500, 800, 1000, 1500, 2000, 3000, 5000]
    dist = con.execute(f"""
        SELECT
            {", ".join(f"SUM(CASE WHEN LENGTH(content) <= {t} THEN 1 ELSE 0 END) AS le{t}" for t in thresholds)},
            COUNT(*) AS total
        FROM news_raw
        WHERE title IS NOT NULL AND content IS NOT NULL
    """).fetchone()

    if dist:
        *counts, total_with_content = dist
        dist_table = Table(show_lines=True)
        dist_table.add_column("Truncation at (chars)", style="yellow")
        dist_table.add_column("Rows fully covered", justify="right", style="green")
        dist_table.add_column("Coverage %", justify="right")
        for threshold, count in zip(thresholds, counts):
            pct = count / total_with_content * 100 if total_with_content else 0
            dist_table.add_row(f"≤ {threshold:,}", f"{count:,}", f"{pct:.1f}%")
        console.print(dist_table)

    con.close()


if __name__ == "__main__":
    main()
