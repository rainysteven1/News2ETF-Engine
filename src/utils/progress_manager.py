"""Progress manager for tracking and publishing labeling pipeline progress via Redis Streams.

Usage:
    manager = ProgressManager(run_id)
    manager.init_major("科技", total_batches=20)
    manager.update_progress("科技", batch_idx=5, saved_count=150, tokens=50000, elapsed=12.5)
    manager.finalize("科技", total_saved=1000, total_tokens=500000, total_time=120.5)
"""

import threading
import time

from pydantic import BaseModel

from src.utils.redis_progress import publish_progress


class MajorState(BaseModel):
    """Internal state for a single major category batch."""

    total_batches: int = 0
    batch_idx: int = 0
    saved_count: int = 0
    tokens: int = 0
    elapsed: float = 0.0
    is_init: bool = False
    is_done: bool = False


class ProgressManager:
    """Manages progress state for a single run across all major categories.

    Tracks per-major progress and publishes updates to Redis SSE stream.
    Thread-safe for use with ThreadPoolExecutor in level-2 parallel mode.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._start_time = time.time()
        # major -> list of MajorState (one per batch update)
        self._states: dict[str, list[MajorState]] = {}
        self._overall = MajorState()
        self._lock = threading.RLock()

    def init_overall(self, total_batches: int) -> None:
        """Publish init for the overall aggregated stream (major="")."""
        self._overall = MajorState(total_batches=total_batches, is_init=True)
        self._publish(
            type="init",
            total_batches=total_batches,
        )

    def init_major(self, major: str, total_batches: int) -> None:
        """Initialize progress tracking for a major category."""
        with self._lock:
            self._states[major] = [MajorState(total_batches=total_batches, is_init=True)]
            self._publish(
                major=major,
                type="init",
                total_batches=total_batches,
            )

    def update_progress(
        self,
        major: str,
        batch_idx: int,
        saved_count: int,
        tokens: int,
        elapsed: float,
        error: str | None = None,
    ) -> None:
        """Append a progress update for a major category."""
        with self._lock:
            if major not in self._states:
                return
            state = MajorState(
                total_batches=self._states[major][0].total_batches if self._states[major] else 0,
                batch_idx=batch_idx,
                saved_count=saved_count,
                tokens=tokens,
                elapsed=elapsed,
            )
            self._states[major].append(state)

            # Update overall sums
            self._overall.batch_idx += 1
            self._overall.saved_count = sum(s[-1].saved_count for s in self._states.values() if s and s[-1].is_done)
            self._overall.tokens = sum(s[-1].tokens for s in self._states.values() if s)

            # Publish per-major progress
            self._publish(
                major=major,
                type="progress",
                batch_idx=batch_idx,
                saved_count=saved_count,
                tokens=tokens,
                elapsed=elapsed,
                error=error,
            )

            # Publish overall progress
            self._publish(
                type="progress",
                saved_count=self._overall.saved_count,
                tokens=self._overall.tokens,
            )

    def finalize(self, major: str, total_saved: int, total_tokens: int, total_time: float) -> None:
        """Mark a major category as done and publish final stats."""
        with self._lock:
            if major not in self._states:
                return

            # Mark last state as done
            if self._states[major]:
                last = self._states[major][-1]
                done_state = MajorState(
                    total_batches=last.total_batches,
                    batch_idx=last.batch_idx,
                    saved_count=total_saved,
                    tokens=total_tokens,
                    elapsed=total_time,
                    is_done=True,
                )
                self._states[major].append(done_state)

            # Update overall sums
            self._overall.saved_count = sum(s[-1].saved_count for s in self._states.values() if s and s[-1].is_done)
            self._overall.tokens = sum(s[-1].tokens for s in self._states.values() if s)

            last = self._states[major][-1]
            self._publish(
                major=major,
                type="done",
                saved_count=total_saved,
                tokens=total_tokens,
                elapsed=total_time,
            )

            # If all majors are done, publish overall done with total elapsed time
            if all(s[-1].is_done for s in self._states.values() if s):
                self._overall.elapsed = time.time() - self._start_time
                self._publish(
                    major="",
                    type="done",
                    saved_count=self._overall.saved_count,
                    tokens=self._overall.tokens,
                    elapsed=self._overall.elapsed,
                )

    def _publish(
        self,
        type: str,
        *,
        major: str | None = None,
        total_batches: int | None = None,
        batch_idx: int | None = None,
        saved_count: int | None = None,
        tokens: int | None = None,
        elapsed: float | None = None,
        error: str | None = None,
    ) -> None:
        """Publish a progress update to Redis SSE stream."""
        raw_list = [("type", type)]
        if total_batches is not None:
            raw_list.append(("total_batches", str(total_batches)))
        if batch_idx is not None:
            raw_list.append(("batch_idx", str(batch_idx)))
        if saved_count is not None:
            raw_list.append(("saved_count", str(saved_count)))
        if tokens is not None:
            raw_list.append(("tokens", str(tokens)))
        if elapsed is not None:
            raw_list.append(("elapsed", f"{elapsed:.2f}"))
        if error is not None:
            raw_list.append(("error", error))

        data = dict(raw_list)
        publish_progress(run_id=self.run_id, major=major, data=data)
