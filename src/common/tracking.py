"""
Generic SQLite tracking decorator.

Records function parameters and results as a single row in a SQLite table.
Works with any pair of Pydantic models (params + result).
"""

import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from src.utils.id_gen import generate_batch_id

T = TypeVar("T")

TRACKING_DB = Path(__file__).resolve().parent.parent.parent / "data" / "tracking.db"


def _ensure_table(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Create table if not exists; add missing columns via ALTER TABLE."""
    col_defs = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})")

    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, dtype in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {dtype}")


def _python_type_to_sqlite(val: object) -> str:
    if isinstance(val, int):
        return "INTEGER"
    if isinstance(val, float):
        return "REAL"
    return "TEXT"


def _build_columns(params: BaseModel | None, result: BaseModel | None) -> dict[str, str]:
    """Build column name → SQLite type mapping from Pydantic models + metadata fields."""
    columns: dict[str, str] = {
        "run_id": "TEXT",
        "started_at": "TEXT",
        "finished_at": "TEXT",
        "elapsed_sec": "REAL",
        "status": "TEXT",
        "error": "TEXT",
    }
    for model in (params, result):
        if model is None:
            continue
        for field_name, value in model.model_dump().items():
            if field_name not in columns:
                columns[field_name] = _python_type_to_sqlite(value)
    return columns


def track(
    table: str,
    params_cls: type[BaseModel],
    result_cls: type[BaseModel],
    db_path: Path = TRACKING_DB,
) -> Callable:
    """
    Decorator factory: log params + result to SQLite.

    The decorated function must return an instance of ``result_cls``.
    The first positional argument after any DI-injected objects must be
    convertible to ``params_cls``, OR keyword arguments must match
    ``params_cls`` fields — the caller decides how to provide params.

    Usage::

        @track("label_runs", LabelParams, LabelResult)
        def label(params: LabelParams, ...) -> LabelResult:
            ...
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(params: BaseModel, *args: object, **kwargs: object) -> T:
            started_at = datetime.now(UTC).isoformat()
            t0 = time.perf_counter()

            # Generate a short readable run ID, pre-insert a "running" row
            run_id: str = generate_batch_id()
            dummy_result = result_cls.model_construct()
            columns = _build_columns(params, dummy_result)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            try:
                _ensure_table(conn, table, columns)
                init_row: dict[str, object] = {
                    "run_id": run_id,
                    "started_at": started_at,
                    "status": "running",
                }
                init_row.update(params.model_dump())
                placeholders = ", ".join("?" for _ in init_row)
                col_names = ", ".join(init_row.keys())
                conn.execute(
                    f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})",
                    list(init_row.values()),
                )
                conn.commit()
            except Exception:
                pass
            finally:
                conn.close()

            # Inject run_id so the wrapped function can use it
            kwargs["run_id"] = run_id

            status = "success"
            error_msg: str | None = None
            result_obj: BaseModel | None = None

            try:
                result_obj = fn(params, *args, **kwargs)  # type: ignore[assignment]
                return result_obj  # type: ignore[return-value]
            except Exception as exc:
                status = "error"
                error_msg = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed = time.perf_counter() - t0
                finished_at = datetime.now(UTC).isoformat()

                update_row: dict[str, object] = {
                    "finished_at": finished_at,
                    "elapsed_sec": round(elapsed, 3),
                    "status": status,
                    "error": error_msg,
                }
                if result_obj is not None:
                    update_row.update(result_obj.model_dump())

                conn = sqlite3.connect(str(db_path))
                try:
                    set_clause = ", ".join(f"{k} = ?" for k in update_row)
                    conn.execute(
                        f"UPDATE {table} SET {set_clause} WHERE run_id = ?",
                        [*update_row.values(), run_id],
                    )
                    conn.commit()
                except Exception:
                    pass
                finally:
                    conn.close()

        return wrapper  # type: ignore[return-value]

    return decorator


def save_checkpoint(
    run_id: str | None,
    stage: str,
    batch_idx: int,
    saved_count: int,
    db_path: Path = TRACKING_DB,
) -> None:
    """Record a labeling checkpoint in SQLite tracking DB."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS label_checkpoints (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     TEXT,
                stage      TEXT,
                batch_idx  INTEGER,
                saved_count INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "INSERT INTO label_checkpoints (run_id, stage, batch_idx, saved_count) VALUES (?, ?, ?, ?)",
            (run_id or "unknown", stage, batch_idx, saved_count),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()
