"""
ClickHouse data store management for news data (Singleton).

Stores:
  - ClickHouseStore: manages news_raw and news_classified tables
"""

import polars as pl
from clickhouse_connect import get_client
from loguru import logger
from rich.console import Console

from src.common import DATA_DIR, get_config

console = Console()


class ClickHouseStore:
    """Manages ClickHouse operations for news data (Singleton)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        ch_cfg = get_config().clickhouse
        self._host = ch_cfg.host
        self._port = ch_cfg.port
        self._database = ch_cfg.database
        self._user = ch_cfg.user
        self._password = ch_cfg.password
        self._initialized = True

    def _get_client(self):
        """Get a ClickHouse client."""
        return get_client(
            host=self._host,
            port=self._port,
            database=self._database,
            username=self._user,
            password=self._password,
        )

    def execute(self, query: str, params: dict | None = None):
        """Execute a query and return results as list of tuples."""
        client = self._get_client()
        result = client.query(query, parameters=params)
        return result.result_rows

    def init_db(self) -> None:
        """Create core tables if they do not exist using ReplacesMergeTree engine."""
        client = self._get_client()

        # news_raw table
        client.command("""
            CREATE TABLE IF NOT EXISTS news_raw (
                news_id  String,
                title    String,
                content  String,
                datetime String,
                source   String
            ) ENGINE = MergeTree()
            ORDER BY news_id
        """)

        # news_classified table with ReplacingMergeTree for idempotent writes
        client.command("""
            CREATE TABLE IF NOT EXISTS news_classified (
                news_id        String,
                title          String,
                major_category String,
                confidence     Float32,
                label_source   String,
                task_id        String,
                run_id         String,
                created_at     DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree
            ORDER BY (news_id, task_id)
        """)

        # news_sub_classified table with ReplacingMergeTree for idempotent writes
        client.command("""
            CREATE TABLE IF NOT EXISTS news_sub_classified (
                news_id          String,
                title            String,
                datetime         String,
                major_category   String,
                sub_category     String,
                sentiment        String,
                impact_score     Float32,
                analysis_logic   String,
                key_evidence     String,
                expectation      String,
                confidence       Float32,
                label_source     String,
                level1_task_id   String,
                level2_task_id   String,
                run_id           String,
                created_at       DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree
            ORDER BY (news_id, level2_task_id)
        """)

    def ensure_news_loaded(self) -> int:
        """Import parquet files into news_raw on first run; return total row count."""
        client = self._get_client()

        count = client.query("SELECT COUNT(*) FROM news_raw").result_rows[0][0]
        if count > 0:
            return count

        logger.info("First run: importing news data into ClickHouse...")
        parquet_dir = DATA_DIR / "converted"

        for f in sorted(parquet_dir.glob("tushare_news_*.parquet")):
            logger.info(f"  importing {f.name}...")
            df = pl.read_parquet(f)
            df = df.filter(pl.col("datetime").is_not_null() | pl.col("title").is_not_null())
            df = df.with_columns(
                (pl.col("datetime").fill_null("") + pl.col("title").fill_null("")).hash().cast(pl.Utf8).alias("news_id")
            )
            records = df.to_dicts()
            if records:
                client.insert("news_raw", records)

        count = client.query("SELECT COUNT(*) FROM news_raw").result_rows[0][0]
        logger.info(f"  imported {count} records")
        return count

    def sample_unlabeled(self, n: int) -> pl.DataFrame:
        """Return up to n news rows not yet labeled by any task, ordered by datetime."""
        result = self.execute(f"""
            SELECT r.news_id, r.title
            FROM news_raw r
            LEFT JOIN news_classified c ON r.news_id = c.news_id
            WHERE c.news_id IS NULL
              AND r.title IS NOT NULL
              AND length(r.title) > 5
            ORDER BY r.datetime, r.news_id
            LIMIT {n}
        """)
        if not result:
            return pl.DataFrame()
        return pl.DataFrame(result, schema=["news_id", "title"])

    def sample_news(self, n: int, seed: int | None = None, offset: int = 0) -> pl.DataFrame:
        """Sample n rows from news_raw.

        When *seed* is provided the sample is deterministic (same rows every
        run), which is required for ablation experiments so all tasks operate
        on identical data.  Without a seed the rows are ordered by datetime.

        *offset* skips the first N rows (datetime order) and is only applied
        when *seed* is None — useful for continuation tasks that resume after
        a partial run.
        """
        if seed is not None:
            result = self.execute(f"""
                SELECT news_id, title
                FROM news_raw
                WHERE title IS NOT NULL AND length(title) > 5
                SAMPLE {n}
            """)
        else:
            result = self.execute(f"""
                SELECT news_id, title
                FROM news_raw
                WHERE title IS NOT NULL AND length(title) > 5
                ORDER BY datetime, news_id
                LIMIT {n} OFFSET {offset}
            """)
        if not result:
            return pl.DataFrame()
        return pl.DataFrame(result, schema=["news_id", "title"])

    def save_labels(self, labels: list[dict], run_id: str | None = None) -> int:
        """Insert level-1 label rows into news_classified; idempotent via ReplacesMergeTree."""
        if not labels:
            return 0
        from datetime import datetime

        records = [
            {
                "news_id": r["news_id"],
                "title": r["title"],
                "major_category": r["major_category"],
                "confidence": r["confidence"],
                "label_source": r["label_source"],
                "task_id": r["task_id"],
                "run_id": run_id or "",
                "created_at": datetime.now(),
            }
            for r in labels
        ]
        client = self._get_client()
        df = pl.DataFrame(records)
        client.insert("news_classified", df.to_numpy().tolist(), column_names=df.columns)
        return len(labels)

    def save_sub_labels(self, labels: list[dict], run_id: str | None = None) -> int:
        """Insert level-2 label rows into news_sub_classified; idempotent via ReplacesMergeTree."""
        if not labels:
            return 0
        from datetime import datetime

        records = [
            {
                "news_id": r["news_id"],
                "title": r["title"],
                "datetime": r.get("datetime", ""),
                "major_category": r["major_category"],
                "sub_category": r["sub_category"],
                "sentiment": r.get("sentiment", "中性"),
                "impact_score": r.get("impact_score", 0.5),
                "analysis_logic": r.get("analysis_logic", ""),
                "key_evidence": r.get("key_evidence", ""),
                "expectation": r.get("expectation", "符合预期"),
                "confidence": r.get("confidence", 0.5),
                "label_source": r.get("label_source", ""),
                "level1_task_id": r.get("level1_task_id", ""),
                "level2_task_id": r.get("level2_task_id", ""),
                "run_id": run_id or "",
                "created_at": datetime.now(),
            }
            for r in labels
        ]
        client = self._get_client()
        df = pl.DataFrame(records)
        client.insert("news_sub_classified", df.to_numpy().tolist(), column_names=df.columns)
        return len(labels)

    def export_training_data(
        self,
        level2_task_id: str,
        confidence_threshold: float = 0.8,
    ) -> pl.DataFrame:
        """Export high-confidence level-2 labels for FinBERT training.

        Returns rows with confidence >= threshold, ordered by datetime.
        Typical use: call after a level-2 task completes, then save to Parquet.
        """
        result = self.execute(
            """
            SELECT news_id, title, datetime, major_category, sub_category,
                   sentiment, confidence, label_source
            FROM news_sub_classified
            WHERE level2_task_id = %(task_id)s
              AND confidence >= %(threshold)s
            ORDER BY datetime, news_id
            """,
            {"task_id": level2_task_id, "threshold": confidence_threshold},
        )
        if not result:
            return pl.DataFrame()
        return pl.DataFrame(
            result,
            schema=[
                "news_id",
                "title",
                "datetime",
                "major_category",
                "sub_category",
                "sentiment",
                "confidence",
                "label_source",
            ],
        )


# Global singleton instance
store = ClickHouseStore()
