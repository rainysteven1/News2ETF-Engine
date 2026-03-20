"""Data management endpoints."""

from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import DataConvertResponse, DataLabelsResponse
from src.common import DATA_DIR
from src.db.store import duckdb_store

router = APIRouter()


@router.post("/convert", response_model=DataConvertResponse)
def convert_to_parquet() -> DataConvertResponse:
    """Convert CSV/XLSX files in data/ to Parquet format."""
    output_dir = DATA_DIR / "converted"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(DATA_DIR.glob("*.csv"))
    xlsx_files = list(DATA_DIR.glob("*.xlsx"))

    if not csv_files and not xlsx_files:
        return DataConvertResponse(files_converted=0, message="No CSV or XLSX files found.")

    converted = 0
    for f in csv_files:
        df = pl.read_csv(f)
        df.write_parquet(output_dir / f"{f.stem}.parquet")
        converted += 1

    for f in xlsx_files:
        try:
            import fastexcel

            wb = fastexcel.read_excel(f)
            for sheet_name in wb.sheet_names:
                df = pl.read_excel(f, sheet_name=sheet_name)
                safe = sheet_name.replace("/", "_").replace("\\", "_")
                df.write_parquet(output_dir / f"{f.stem}_{safe}.parquet")
            converted += 1
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to convert {f.name}: {exc}")

    return DataConvertResponse(files_converted=converted, message=f"Done. {converted} file(s) converted.")


@router.get("/labels", response_model=DataLabelsResponse)
def get_labels(
    task_id: str | None = Query(None, description="Filter by task ID (short prefix allowed)"),
    level: int | None = Query(None, description="1 = major only, 2 = sub-category present"),
    limit: int = Query(20, ge=1, le=1000, description="Max records to return"),
) -> DataLabelsResponse:
    """Query labeled news records from DuckDB."""
    if not duckdb_store.db_path.exists():
        raise HTTPException(status_code=503, detail="DuckDB not initialised. Run POST /data/convert first.")

    con = duckdb_store.connect(read_only=True)
    try:
        conditions: list[str] = ["task_id IS NOT NULL"]
        bind_params: list[Any] = []

        if task_id:
            conditions.append("task_id LIKE ?")
            bind_params.append(f"{task_id}%")

        if level == 1:
            conditions.append("sub_category IS NULL")
        elif level == 2:
            conditions.append("sub_category IS NOT NULL")

        where_clause = " AND ".join(conditions)
        query = f"SELECT * FROM news_classified WHERE {where_clause} ORDER BY created_at DESC LIMIT ?"
        bind_params.append(limit)

        df = con.execute(query, bind_params).pl()
    finally:
        con.close()

    records: list[dict[str, Any]] = df.to_dicts()
    return DataLabelsResponse(total=len(records), records=records)
