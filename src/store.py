"""
DuckDB store — table initialization, data ingestion, and label I/O.

All functions operate on an open DuckDBPyConnection passed by the caller.
"""

import duckdb
import polars as pl
from rich.console import Console

from src.common import DATA_DIR

DB_PATH = DATA_DIR / "news2etf.duckdb"


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    """Create core tables if they do not exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_raw (
            news_id  VARCHAR PRIMARY KEY,
            title    VARCHAR,
            content  VARCHAR,
            datetime VARCHAR,
            source   VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_classified (
            news_id        VARCHAR PRIMARY KEY,
            title          VARCHAR,
            major_category VARCHAR,
            sub_category   VARCHAR,
            sentiment      VARCHAR,
            confidence     FLOAT,
            label_source   VARCHAR
        )
    """)


def ensure_news_loaded(con: duckdb.DuckDBPyConnection, console: Console) -> int:
    """Import parquet files into news_raw on first run; return total row count."""
    count = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]  # type: ignore
    if count > 0:
        return count

    console.print("[bold cyan]First run: importing news data into DuckDB...[/bold cyan]")
    parquet_dir = DATA_DIR / "converted"
    for f in sorted(parquet_dir.glob("tushare_news_*.parquet")):
        console.print(f"  importing [cyan]{f.name}[/cyan]...")
        con.execute(f"""
            INSERT OR IGNORE INTO news_raw
            SELECT
                md5(COALESCE(datetime, '') || COALESCE(title, '')) AS news_id,
                title, content, datetime, source
            FROM read_parquet('{f}')
            WHERE datetime IS NOT NULL OR title IS NOT NULL
        """)

    count = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]  # type: ignore
    console.print(f"  imported [bold]{count}[/bold] records")
    return count


def sample_unlabeled(con: duckdb.DuckDBPyConnection, n: int) -> pl.DataFrame:
    """Return up to n unlabeled news rows (news_id, title), ordered by datetime."""
    return con.execute(f"""
        SELECT r.news_id, r.title
        FROM news_raw r
        LEFT JOIN news_classified c ON r.news_id = c.news_id
        WHERE c.news_id IS NULL
          AND r.title IS NOT NULL
          AND LENGTH(r.title) > 5
        ORDER BY r.datetime, r.news_id
        LIMIT {n}
    """).pl()


def save_labels(con: duckdb.DuckDBPyConnection, labels: list[dict]) -> int:
    """Insert new label rows; silently skip duplicates."""
    if not labels:
        return 0
    df = pl.DataFrame(labels)  # noqa: F841
    con.execute("INSERT OR IGNORE INTO news_classified SELECT * FROM df")
    return len(labels)


def batch_update_labels(con: duckdb.DuckDBPyConnection, updates: list[dict]) -> int:
    """Bulk UPDATE via temp table + JOIN instead of per-row UPDATE."""
    if not updates:
        return 0
    df = pl.DataFrame(updates)  # noqa: F841
    con.execute("CREATE OR REPLACE TEMPORARY TABLE _label_updates AS SELECT * FROM df")
    con.execute("""
        UPDATE news_classified
        SET sub_category = u.sub_category,
            sentiment    = u.sentiment,
            confidence   = u.confidence,
            label_source = u.label_source
        FROM _label_updates u
        WHERE news_classified.news_id = u.news_id
    """)
    con.execute("DROP TABLE IF EXISTS _label_updates")
    return len(updates)
