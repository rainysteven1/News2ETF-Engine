#!/usr/bin/env python3
"""Query news_sub_classified table from ClickHouse with confidence > 0.7 filter.

Streams data from ClickHouse and converts to Parquet using Polars.
"""

import tomllib
from pathlib import Path

import clickhouse_connect
import polars as pl


def load_config(config_path: str = "config.dev.toml") -> dict:
    """Load ClickHouse config from TOML file."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config["clickhouse"]


def stream_clickhouse_results(
    task_id: str,
    run_id: str,
    config_path: str = "config.dev.toml",
    confidence_threshold: float = 0.7,
):
    """Generator that streams rows from ClickHouse one at a time.

    Args:
        task_id: The level2_task_id to filter by
        run_id: The run_id to filter by
        config_path: Path to config file
        confidence_threshold: Minimum confidence (default 0.7)

    Yields:
        Tuple of (news_id, title, datetime, major_category, sub_category, sentiment, content)
    """
    ch_config = load_config(config_path)

    client = clickhouse_connect.get_client(
        host=ch_config["host"],
        port=ch_config["port"],
        database=ch_config["database"],
        username=ch_config["user"],
        password=ch_config["password"],
    )

    query = """
        SELECT
            sc.news_id,
            sc.title,
            sc.datetime,
            sc.major_category,
            sc.sub_category,
            sc.sentiment,
            nr.content
        FROM news_sub_classified AS sc
        INNER JOIN news_raw AS nr ON sc.news_id = nr.news_id
        WHERE sc.level2_task_id = %(task_id)s
          AND sc.run_id = %(run_id)s
          AND sc.confidence > %(confidence_threshold)s
        ORDER BY sc.datetime ASC
    """

    result = client.query(
        query,
        parameters={
            "task_id": task_id,
            "run_id": run_id,
            "confidence_threshold": confidence_threshold,
        },
    )

    yield from result.result_rows


def stream_to_parquet(
    task_id: str,
    run_id: str,
    config_path: str = "config.dev.toml",
    confidence_threshold: float = 0.7,
    output_path: str | None = None,
    batch_size: int = 10000,
):
    """Stream query results from ClickHouse and write to Parquet in batches.

    Uses a generator to stream rows from ClickHouse and accumulates them
    in batches before writing to parquet for memory efficiency.

    Args:
        task_id: The level2_task_id to filter by
        run_id: The run_id to filter by
        config_path: Path to config file
        confidence_threshold: Minimum confidence (default 0.7)
        output_path: Output parquet path (default: level2_{task_id}_{run_id}.parquet)
        batch_size: Number of rows to accumulate before writing (default 10000)

    Returns:
        Polars DataFrame with all loaded data
    """
    schema = {
        "news_id": pl.Utf8,
        "title": pl.Utf8,
        "datetime": pl.Utf8,
        "major_category": pl.Utf8,
        "sub_category": pl.Utf8,
        "sentiment": pl.Utf8,
        "content": pl.Utf8,
    }

    if output_path is None:
        output_path = f"level2_{task_id}_{run_id}.parquet"

    batch: list[tuple] = []
    total_rows = 0

    for row in stream_clickhouse_results(task_id, run_id, config_path, confidence_threshold):
        batch.append(row)  # type: ignore
        if len(batch) >= batch_size:
            df = pl.DataFrame(batch, schema=schema)
            if total_rows == 0:
                df.write_parquet(output_path)
            else:
                # Append using pyarrow for efficiency
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pa.Table.from_pandas(df.to_pandas())
                with pq.ParquetWriter(output_path, table.schema, append=True) as writer:
                    writer.write_table(table)
            total_rows += len(batch)
            batch = []

    # Write remaining rows
    if batch:
        df = pl.DataFrame(batch, schema=schema, orient="row")
        if total_rows == 0:
            df.write_parquet(output_path)
            total_rows = len(batch)
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df.to_pandas())
            with pq.ParquetWriter(output_path, table.schema, append=True) as writer:
                writer.write_table(table)
            total_rows += len(batch)

    print(f"Total rows written: {total_rows}")
    return pl.read_parquet(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <task_id> <run_id>")
        print(f"Example: {sys.argv[0]} abc123 def456")
        sys.exit(1)

    task_id = sys.argv[1]
    run_id = sys.argv[2]

    parquet_path = f"level2_{task_id}_{run_id}.parquet"

    df = stream_to_parquet(task_id, run_id, output_path=parquet_path)

    if df.is_empty():
        print("No results found.")
        Path(parquet_path).unlink(missing_ok=True)
    else:
        print(f"\nParquet saved to: {parquet_path}")
        print(f"Shape: {df.shape}")
        print("\nPreview:")
        print(df.head(10))
