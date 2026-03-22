import asyncio
from typing import Any
from urllib.parse import quote

import redis
from loguru import logger
from redis.asyncio import Redis

from src.common import get_config

_redis_client: redis.Redis | None = None
_async_redis_client: Redis | None = None


def _get_redis_client() -> redis.Redis:
    """Get or create a singleton sync Redis client."""
    global _redis_client
    if _redis_client is None:
        cfg = get_config().redis
        _redis_client = redis.Redis(host=cfg.host, port=cfg.port, decode_responses=True)
    return _redis_client


async def _get_async_redis_client() -> Redis:
    """Get or create a singleton async Redis client."""
    global _async_redis_client
    if _async_redis_client is None:
        cfg = get_config().redis
        _async_redis_client = Redis(host=cfg.host, port=cfg.port, decode_responses=True)
    return _async_redis_client


def publish_progress(run_id: str, data: dict[str, str], major: str | None = None) -> None:
    """Publish a progress message to the Redis Stream.

    Args:
        run_id: The 8-char run identifier.
        major: The major category name (use "" for overall/aggregated progress).
               Will be URL-encoded for storage safety.
        data: The progress data dict to publish.
    """
    if major:
        encoded_major = quote(major, safe="")
        key = f"sse:progress:{run_id}:{encoded_major}"
    else:
        key = f"sse:progress:{run_id}"

    try:
        client = _get_redis_client()
        client.xadd(key, data)  # type: ignore
    except Exception as e:
        logger.error(f"Failed to publish progress to Redis for run_id={run_id}, major={major}: {e}")


async def iterate_progress(
    run_id: str,
    major: str | None = None,
    timeout_ms: int = 30000,
):
    """Async generator that yields progress messages from Redis Stream.

    Reads historical messages first (from "0"), then switches to "$" for new messages.
    Stops early if a "done" message is encountered.

    Args:
        run_id: The 8-char run identifier.
        major: If provided, only listen to this major category's stream.
               If None, listen to the aggregated stream (empty major = "").
        timeout_ms: Blocking timeout in milliseconds.

    Yields:
        tuples of (message_id, progress_data_dict)
    """
    if major:
        encoded_major = quote(major, safe="")
        key = f"sse:progress:{run_id}:{encoded_major}"
    else:
        key = f"sse:progress:{run_id}"

    client = await _get_async_redis_client()
    last_id = "0"  # Start from beginning to read historical data

    try:
        while True:
            try:
                result = await client.xread({key: last_id}, block=timeout_ms)

                if not result:
                    # No more historical data, switch to listening for new messages
                    if last_id == "0":
                        last_id = "$"
                        continue
                    break

                for _, messages in result:
                    for msg_id, msg_data in messages:
                        last_id = msg_id
                        data = dict(msg_data)
                        yield msg_id, data
                        # If this is a "done" message, stop and let client disconnect
                        if data.get("type") == "done":
                            return

                # After exhausting historical data, switch to new messages
                if last_id == "0":
                    last_id = "$"
            except Exception:
                await asyncio.sleep(1)
                continue
    finally:
        await client.aclose()
