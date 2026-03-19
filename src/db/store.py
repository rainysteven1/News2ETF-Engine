"""
Unified data store management for DuckDB using singleton pattern.

Stores:
  - DuckDBStore: manages news_raw and news_classified tables
  - PostgreSQL (via SQLAlchemy): manages experiments, tasks, configs, and checkpoints (see db/models.py)
"""

from pathlib import Path

import duckdb
import polars as pl
from rich.console import Console

from src.common import DATA_DIR

console = Console()


class DuckDBStore:
    """Manages DuckDB operations for news data (Singleton)."""

    _instance = None

    def __new__(cls, db_path: Path | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Path | None = None):
        if self._initialized:
            return
        self.db_path = db_path or DATA_DIR / "news2etf.duckdb"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def connect(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        """Create a new connection."""
        return duckdb.connect(str(self.db_path), read_only=read_only)

    def init_db(self) -> None:
        """Create core tables if they do not exist."""
        con = self.connect()
        try:
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
                    news_id        VARCHAR,
                    title          VARCHAR,
                    major_category VARCHAR,
                    sub_category   VARCHAR,
                    sentiment      VARCHAR,
                    confidence     FLOAT,
                    label_source   VARCHAR,
                    task_id        VARCHAR,
                    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (news_id, task_id)
                )
            """)
        finally:
            con.close()

    def ensure_news_loaded(self) -> int:
        """Import parquet files into news_raw on first run; return total row count."""
        con = self.connect()
        try:
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
        finally:
            con.close()

    def sample_unlabeled(self, con: duckdb.DuckDBPyConnection, n: int) -> pl.DataFrame:
        """Return up to n news rows not yet labeled by any task, ordered by datetime."""
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

    def sample_news(self, con: duckdb.DuckDBPyConnection, n: int, seed: int | None = None) -> pl.DataFrame:
        """Sample n rows from news_raw.

        When *seed* is provided the sample is deterministic (same rows every
        run), which is required for ablation experiments so all tasks operate
        on identical data.  Without a seed the rows are ordered by datetime.
        """
        if seed is not None:
            return con.execute(f"""
                SELECT news_id, title
                FROM (
                    SELECT news_id, title
                    FROM news_raw
                    WHERE title IS NOT NULL AND LENGTH(title) > 5
                ) USING SAMPLE RESERVOIR({n} ROWS) REPEATABLE ({seed})
            """).pl()
        return con.execute(f"""
            SELECT news_id, title
            FROM news_raw
            WHERE title IS NOT NULL AND LENGTH(title) > 5
            ORDER BY datetime, news_id
            LIMIT {n}
        """).pl()

    def save_labels(self, con: duckdb.DuckDBPyConnection, labels: list[dict]) -> int:
        """Insert new label rows; silently skip duplicates."""
        if not labels:
            return 0
        df = pl.DataFrame(labels)  # noqa: F841
        con.execute("""
            INSERT OR IGNORE INTO news_classified
                (news_id, title, major_category, sub_category, sentiment, confidence, label_source, task_id)
            SELECT news_id, title, major_category, sub_category, sentiment, confidence, label_source, task_id
            FROM df
        """)
        return len(labels)

    def batch_update_labels(self, con: duckdb.DuckDBPyConnection, updates: list[dict]) -> int:
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
              AND news_classified.task_id = u.task_id
        """)
        con.execute("DROP TABLE IF EXISTS _label_updates")
        return len(updates)


# Global singleton instance
duckdb_store = DuckDBStore()
