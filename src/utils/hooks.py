"""
SQLite Hook Registry — APSW-based update hooks for experiment status tracking.

When an experiment's status changes to 'completed' or 'failed', registered
callbacks fire so the runner can count completions and decide what to do next
(e.g. start the next experiment in a batch, or move to the next batch).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import apsw

logger = logging.getLogger(__name__)

# Hook callback signature: (experiment_id, new_status, extra_context) -> None
HookCallback = Callable[[str, str, dict[str, Any]], None]


@dataclass
class _HookEntry:
    name: str
    callback: HookCallback
    on_status: set[str]  # statuses to trigger on, e.g. {"completed", "failed"}


class SQLiteHookRegistry:
    """
    Watches the experiments table for UPDATE operations via APSW's
    update_hook and dispatches registered callbacks when status changes
    to a terminal state (completed / failed).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._hooks: list[_HookEntry] = []
        self._conn: apsw.Connection | None = None

    # ---- connection management ------------------------------------------------

    def connect(self) -> apsw.Connection:
        """Open (or return existing) APSW connection with the update hook installed."""
        if self._conn is None:
            self._conn = apsw.Connection(self.db_path)
            self._conn.setupdatehook(self._on_update)
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---- hook registration ----------------------------------------------------

    def register(
        self,
        name: str,
        callback: HookCallback,
        on_status: set[str] | None = None,
    ):
        """
        Register a callback that fires when experiment status changes.

        Args:
            name: Human-readable label for logging.
            callback: (experiment_id, new_status, row_dict) -> None
            on_status: Set of statuses to trigger on. Default: {"completed", "failed"}
        """
        if on_status is None:
            on_status = {"completed", "failed"}
        self._hooks.append(_HookEntry(name=name, callback=callback, on_status=on_status))

    def unregister(self, name: str):
        """Remove a hook by name."""
        self._hooks = [h for h in self._hooks if h.name != name]

    # ---- APSW update hook -----------------------------------------------------

    def _on_update(self, update_type: int, db_name: str, table_name: str, rowid: int):
        """
        Called by APSW on every INSERT/UPDATE/DELETE.
        We only care about UPDATEs on the experiments table.
        """
        if table_name != "experiments":
            return
        # APSW update types: 18=INSERT, 23=UPDATE, 9=DELETE
        if update_type != 23:
            return

        conn = self.connect()
        try:
            cursor = conn.execute("SELECT * FROM experiments WHERE rowid = ?", (rowid,))
            description = cursor.getdescription()
            col_names = [d[0] for d in description]
            row_tuple = cursor.fetchone()
            if row_tuple is None:
                return
            row = dict(zip(col_names, row_tuple))
        except Exception:
            logger.exception("Failed to read updated row %d", rowid)
            return

        status = row.get("status")
        experiment_id = row.get("experiment_id", "")

        for hook in self._hooks:
            if status in hook.on_status:
                try:
                    hook.callback(experiment_id, status, row)
                except Exception:
                    logger.exception("Hook '%s' raised an exception", hook.name)

    # ---- convenience helpers ---------------------------------------------------

    def execute(self, sql: str, bindings: tuple | list = ()):
        """Execute SQL through the hooked connection."""
        conn = self.connect()
        conn.execute(sql, bindings)

    def fetchone(self, sql: str, bindings: tuple | list = ()) -> dict[str, Any] | None:
        conn = self.connect()
        cursor = conn.execute(sql, bindings)
        description = cursor.getdescription()
        col_names = [d[0] for d in description]
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(zip(col_names, row))

    def fetchall(self, sql: str, bindings: tuple | list = ()) -> list[dict[str, Any]]:
        conn = self.connect()
        cursor = conn.execute(sql, bindings)
        description = cursor.getdescription()
        col_names = [d[0] for d in description]
        return [dict(zip(col_names, r)) for r in cursor.fetchall()]


@dataclass
class BatchProgress:
    """Tracks completion progress for a batch of experiments."""

    batch_id: str
    total: int
    completed: int = 0
    failed: int = 0
    _callbacks_on_batch_done: list[Callable[[str, int, int], None]] = field(default_factory=list)

    @property
    def finished(self) -> int:
        return self.completed + self.failed

    @property
    def remaining(self) -> int:
        return self.total - self.finished

    @property
    def is_done(self) -> bool:
        return self.remaining <= 0

    def on_batch_done(self, callback: Callable[[str, int, int], None]):
        """Register callback(batch_id, completed, failed) when all experiments finish."""
        self._callbacks_on_batch_done.append(callback)

    def record(self, status: str):
        if status == "completed":
            self.completed += 1
        elif status == "failed":
            self.failed += 1
        if self.is_done:
            for cb in self._callbacks_on_batch_done:
                cb(self.batch_id, self.completed, self.failed)


def create_batch_tracker(
    hook_registry: SQLiteHookRegistry,
    batch_id: str,
    experiment_ids: list[str],
    on_experiment_done: Callable[[str, str, int, int], None] | None = None,
    on_batch_done: Callable[[str, int, int], None] | None = None,
) -> BatchProgress:
    """
    Wire up a BatchProgress tracker to the hook registry for a given batch.

    Args:
        hook_registry: The SQLiteHookRegistry instance.
        batch_id: Batch identifier.
        experiment_ids: List of experiment IDs in this batch.
        on_experiment_done: Optional callback(experiment_id, status, completed_so_far, remaining).
        on_batch_done: Optional callback(batch_id, completed_count, failed_count)
            invoked when all experiments in the batch have finished.

    Returns:
        BatchProgress instance being tracked.
    """
    exp_set = set(experiment_ids)
    progress = BatchProgress(batch_id=batch_id, total=len(experiment_ids))

    if on_batch_done:
        progress.on_batch_done(on_batch_done)

    def _hook(experiment_id: str, new_status: str, row: dict[str, Any]):
        if experiment_id not in exp_set:
            return
        progress.record(new_status)
        if on_experiment_done:
            on_experiment_done(experiment_id, new_status, progress.finished, progress.remaining)

    hook_registry.register(
        name=f"batch_tracker_{batch_id}",
        callback=_hook,
        on_status={"completed", "failed"},
    )
    return progress
