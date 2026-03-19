"""
Loguru sink that pushes log entries to Grafana Loki via HTTP push API.

Uses only stdlib (urllib) — no extra dependencies required.
"""

import json
import sys
import time
import urllib.request
from threading import Lock, Thread

from src.common.config import console


class LokiSink:
    """Buffered loguru sink for Loki."""

    def __init__(
        self,
        url: str,
        labels: dict[str, str],
        batch_size: int = 10,
        flush_interval: float = 5.0,
    ):
        self.push_url = url.rstrip("/") + "/loki/api/v1/push"
        self.labels = labels
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._buffer: list[list[str]] = []
        self._lock = Lock()
        self._running = True
        self._flusher = Thread(target=self._flush_loop, daemon=True)
        self._flusher.start()

    def _flush_loop(self) -> None:
        while self._running:
            time.sleep(self.flush_interval)
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            entries = self._buffer.copy()
            self._buffer.clear()

        payload = {
            "streams": [
                {
                    "stream": self.labels,
                    "values": entries,
                }
            ]
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.push_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)  # noqa: S310
        except Exception as exc:
            console.print(f"[bold red]Failed to push logs to Loki:[/bold red] {exc}")

    def write(self, message: str) -> None:
        """Loguru sink interface — receives formatted message string."""
        record = message.record  # type: ignore[attr-defined]
        ts_ns = str(int(record["time"].timestamp() * 1e9))
        line = str(message).rstrip("\n")
        with self._lock:
            self._buffer.append([ts_ns, line])
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def stop(self) -> None:
        """Flush remaining logs and stop the background flusher."""
        self._running = False
        self._flush()
