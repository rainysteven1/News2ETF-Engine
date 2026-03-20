"""
Thin S3 client for SeaweedFS artifact storage.

Provides upload_json() for persisting LLM diagnostic dumps.
Bucket auto-creation is handled transparently.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from loguru import logger

from src.common import get_config


@lru_cache(maxsize=1)
def _get_s3_client():
    cfg = get_config().s3
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key or "any",
        aws_secret_access_key=cfg.secret_key or "any",
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


_ensured_buckets: set[str] = set()


def _ensure_bucket(client, bucket: str) -> None:
    if bucket in _ensured_buckets:
        return
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
    _ensured_buckets.add(bucket)


def upload_json(bucket: str, key: str, data: dict[str, Any]) -> bool:
    """Upload a JSON object to SeaweedFS. Returns True on success."""
    try:
        client = _get_s3_client()
        _ensure_bucket(client, bucket)
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        logger.debug(f"S3 uploaded s3://{bucket}/{key} ({len(body)} bytes)")
        return True
    except Exception as e:
        logger.warning(f"S3 upload failed s3://{bucket}/{key}: {e}")
        return False
