"""
Batch ID, Experiment ID and W&B run name generation utilities.

Batch ID format:     YYYYMMDD_HHMMSS_xxxx   (xxxx = 4 random lowercase letters)
Experiment ID format: {batch_id}_{idx}
W&B run name format:  {experiment_id}_{yyyy}  (yyyy = 4 random alphanumeric chars)
"""

import random
import string
from datetime import datetime


def generate_batch_id(now: datetime | None = None) -> str:
    """
    Generate a unique batch ID: YYYYMMDD_HHMMSS_xxxx

    Args:
        now: Optional datetime for deterministic testing. Defaults to datetime.now().

    Returns:
        Batch ID string, e.g. "20260309_102030_abcd"
    """
    if now is None:
        now = datetime.now()
    suffix = "".join(random.choices(string.ascii_lowercase, k=4))
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{suffix}"


def make_experiment_id(batch_id: str, idx: int) -> str:
    """
    Build an experiment ID from a batch ID and an index.

    Args:
        batch_id: e.g. "20260309_102030_abcd"
        idx: 0-based index within the batch

    Returns:
        Experiment ID string, e.g. "20260309_102030_abcd_0"
    """
    return f"{batch_id}_{idx}"


def parse_experiment_id(experiment_id: str) -> tuple[str, str, int]:
    """
    Parse an experiment ID into (date, batch_id, idx).

    Args:
        experiment_id: e.g. "20260309_102030_abcd_0"

    Returns:
        (date_str, batch_id, idx)
        e.g. ("20260309", "20260309_102030_abcd", 0)
    """
    parts = experiment_id.rsplit("_", 1)
    batch_id = parts[0]
    idx = int(parts[1])
    date_str = batch_id.split("_")[0]
    return date_str, batch_id, idx


def generate_wandb_run_name(experiment_id: str) -> str:
    """
    Generate a unique W&B run name: {experiment_id}_{yyyy}

    A 4-char alphanumeric suffix is appended so that re-running the same
    experiment never collides with a previous W&B run.

    Args:
        experiment_id: e.g. "20260309_102030_abcd_0"

    Returns:
        W&B run name string, e.g. "20260309_102030_abcd_0_x3k9"
    """
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{experiment_id}_{suffix}"
