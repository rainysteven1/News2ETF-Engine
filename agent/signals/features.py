"""Feature engineering — aggregate news sentiment into industry-level time series."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import polars as pl
from loguru import logger

from agent.utils.industry_map import IndustryMapper


def aggregate_industry_sentiment(
    classified_path: Path,
    industry_dict_path: Path,
    output_path: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Aggregate classified news into daily industry-level sentiment features.

    Args:
        classified_path: Path to classified news parquet (with sub_category, sentiment columns)
        industry_dict_path: Path to industry_dict.json
        output_path: Optional path to save the result parquet
        start_date: Optional filter start date (YYYY-MM-DD)
        end_date: Optional filter end date (YYYY-MM-DD)

    Returns:
        DataFrame with columns: industry, date, sentiment_mean, sentiment_std,
        news_count, news_heat, sentiment_trend
    """
    logger.info("Loading industry mapper from {}", industry_dict_path)
    mapper = IndustryMapper(industry_dict_path)

    logger.info("Loading classified news from {}", classified_path)
    df = pl.read_parquet(classified_path)
    logger.info("Loaded {} rows", len(df))

    # Parse datetime column
    df = df.with_columns(
        pl.col("datetime").str.to_datetime().dt.date().alias("date")
    )

    # Filter by date range if provided
    if start_date:
        df = df.filter(pl.col("date") >= Date.fromisoformat(start_date))
    if end_date:
        df = df.filter(pl.col("date") <= Date.fromisoformat(end_date))

    # Map sub_category to industry
    df = df.with_columns(
        pl.col("sub_category").map_dict(
            {sub: mapper.get_industry(sub) for sub in mapper._sub_to_industry},
            default=None,
        ).alias("industry")
    )

    # Drop rows with no industry mapping
    df = df.filter(pl.col("industry").is_not_null())

    # Ensure sentiment is numeric
    df = df.with_columns(
        pl.col("sentiment").cast(pl.Float64)
    )

    # Aggregate by industry and date
    result = df.group_by(["industry", "date"]).agg([
        pl.col("sentiment").mean().alias("sentiment_mean"),
        pl.col("sentiment").std().alias("sentiment_std"),
        pl.col("sentiment").count().alias("news_count"),
    ])

    # Fill std nulls with 0 (single news item)
    result = result.with_columns(
        pl.col("sentiment_std").fill_null(0.0)
    )

    # news_heat: normalized by rolling mean of past 20 days news_count
    result = result.sort(["industry", "date"])

    # Compute news_heat as current news_count / rolling_mean(20)
    result = result.with_columns(
        (pl.col("news_count") / pl.col("news_count").rolling_mean(20).over("industry").fill_null(1.0))
        .clip(0.0, 10.0)
        .alias("news_heat")
    )

    # sentiment_trend: 5-day momentum (current mean - 5-day ago mean)
    sentiment_diff = (
        pl.col("sentiment_mean") - pl.col("sentiment_mean").shift(5).over("industry")
    ).alias("sentiment_trend")

    result = result.with_columns(sentiment_diff)

    # Sort
    result = result.sort(["industry", "date"])

    logger.info("Aggregated {} rows across {} industries", len(result), result["industry"].n_unique())

    if output_path:
        logger.info("Saving industry sentiment to {}", output_path)
        result.write_parquet(output_path)

    return result
