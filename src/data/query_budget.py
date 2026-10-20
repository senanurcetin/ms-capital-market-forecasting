"""Per-job scan ceiling. Account-level quotas remain necessary for a monthly budget."""
import os

from google.cloud import bigquery

DEFAULT_MAX_BYTES = 1_000_000_000


def query_config() -> bigquery.QueryJobConfig:
    """Reject invalid/unlimited overrides before a client can submit a query."""
    value = os.environ.get("MSCAPITAL_BQ_MAX_BYTES", str(DEFAULT_MAX_BYTES))
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError("MSCAPITAL_BQ_MAX_BYTES must be a positive integer") from exc
    if limit <= 0:
        raise ValueError("MSCAPITAL_BQ_MAX_BYTES must be a positive integer")
    return bigquery.QueryJobConfig(maximum_bytes_billed=limit)
