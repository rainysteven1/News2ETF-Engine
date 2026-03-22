"""Labeling-specific endpoints."""

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.utils.redis_progress import iterate_progress

router = APIRouter()


@router.get("/runs/{run_id}/progress")
async def stream_progress(run_id: str, major: str | None = None):
    """SSE endpoint for real-time progress streaming.

    Query params:
    - major: 可选，特定 major 分类的进度，不传则汇总所有

    SSE message format:
    - Init: {"type": "init", "total_batches": int, "major": str}
    - Progress: {"type": "progress", "batch_idx": int, "saved_count": int, "tokens": int, "elapsed": float}
    - Done: {"type": "done", "total_saved": int, "total_tokens": int, "total_time": float}
    """

    async def event_generator():
        try:
            async for msg_id, data in iterate_progress(run_id, major=major, timeout_ms=30000):
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("type") == "done":
                    break
        except Exception:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
