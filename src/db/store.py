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
                    confidence     FLOAT,
                    label_source   VARCHAR,
                    task_id        VARCHAR,
                    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (news_id, task_id)
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS news_sub_classified (
                    news_id          VARCHAR,
                    title            VARCHAR,
                    datetime         VARCHAR,
                    major_category   VARCHAR,
                    sub_category     VARCHAR,
                    sentiment        VARCHAR,
                    impact_score     FLOAT,
                    analysis_logic   VARCHAR,
                    key_evidence     VARCHAR,
                    expectation      VARCHAR,
                    confidence       FLOAT,
                    label_source     VARCHAR,
                    level1_task_id   VARCHAR,
                    level2_task_id   VARCHAR,
                    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (news_id, level1_task_id, level2_task_id)
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

    def sample_news(
        self, con: duckdb.DuckDBPyConnection, n: int, seed: int | None = None, offset: int = 0
    ) -> pl.DataFrame:
        """Sample n rows from news_raw.

        When *seed* is provided the sample is deterministic (same rows every
        run), which is required for ablation experiments so all tasks operate
        on identical data.  Without a seed the rows are ordered by datetime.

        *offset* skips the first N rows (datetime order) and is only applied
        when *seed* is None — useful for continuation tasks that resume after
        a partial run.
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
            LIMIT {n} OFFSET {offset}
        """).pl()

    def save_labels(self, con: duckdb.DuckDBPyConnection, labels: list[dict]) -> int:
        """Insert level-1 label rows into news_classified; silently skip duplicates."""
        if not labels:
            return 0
        df = pl.DataFrame(
            [
                {
                    "news_id": r["news_id"],
                    "title": r["title"],
                    "major_category": r["major_category"],
                    "confidence": r["confidence"],
                    "label_source": r["label_source"],
                    "task_id": r["task_id"],
                }
                for r in labels
            ]
        )  # noqa: F841
        con.execute("""
            INSERT OR IGNORE INTO news_classified
                (news_id, title, major_category, confidence, label_source, task_id)
            SELECT news_id, title, major_category, confidence, label_source, task_id
            FROM df
        """)
        con.commit()
        return len(labels)

    def save_sub_labels(self, con: duckdb.DuckDBPyConnection, labels: list[dict]) -> int:
        """Insert level-2 label rows into news_sub_classified; silently skip duplicates."""
        if not labels:
            return 0
        df = pl.DataFrame(labels)  # noqa: F841
        con.execute("""
            INSERT OR IGNORE INTO news_sub_classified
                (news_id, title, datetime, major_category, sub_category, sentiment,
                 impact_score, analysis_logic, key_evidence, expectation,
                 confidence, label_source, level1_task_id, level2_task_id)
            SELECT news_id, title, datetime, major_category, sub_category, sentiment,
                   impact_score, analysis_logic, key_evidence, expectation,
                   confidence, label_source, level1_task_id, level2_task_id
            FROM df
        """)
        con.commit()
        return len(labels)

    def export_training_data(
        self,
        con: duckdb.DuckDBPyConnection,
        level2_task_id: str,
        confidence_threshold: float = 0.8,
    ) -> pl.DataFrame:
        """Export high-confidence level-2 labels for FinBERT training.

        Returns rows with confidence >= threshold, ordered by datetime.
        Typical use: call after a level-2 task completes, then save to Parquet.
        """
        return con.execute(
            """
            SELECT news_id, title, datetime, major_category, sub_category,
                   sentiment, confidence, label_source
            FROM news_sub_classified
            WHERE level2_task_id = ?
              AND confidence >= ?
            ORDER BY datetime, news_id
            """,
            [level2_task_id, confidence_threshold],
        ).pl()


# Global singleton instance
duckdb_store = DuckDBStore()
