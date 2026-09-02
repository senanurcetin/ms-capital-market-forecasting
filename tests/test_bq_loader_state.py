"""Stale-state protection for the uploader - WITHOUT ANY LIVE BQ CALLS.

The scenario is real: mscapital_raw was dropped for cost reasons after staging was
built, leaving the _loaded.json files behind. Without this guard load_group would
silently upload nothing and then fail with a confusing error.
"""
from google.cloud.exceptions import NotFound

from src.data.bq_loader import reset_state_if_table_missing


class FakeBQ:
    def __init__(self, exists: bool):
        self.exists = exists
        self.calls = 0

    def get_table(self, table_id):
        self.calls += 1
        if not self.exists:
            raise NotFound(table_id)
        return object()


def test_state_reset_when_table_missing():
    state = {"table_id": "p.d.t", "loaded": ["a.parquet", "b.parquet"]}
    out = reset_state_if_table_missing(FakeBQ(exists=False), "p.d.t", state)
    assert out["loaded"] == []


def test_state_kept_when_table_exists():
    state = {"table_id": "p.d.t", "loaded": ["a.parquet"]}
    out = reset_state_if_table_missing(FakeBQ(exists=True), "p.d.t", state)
    assert out["loaded"] == ["a.parquet"]


def test_empty_state_skips_lookup_entirely():
    """With no parts loaded, BigQuery should not be queried at all."""
    bq = FakeBQ(exists=True)
    out = reset_state_if_table_missing(bq, "p.d.t", {"table_id": "p.d.t", "loaded": []})
    assert out["loaded"] == [] and bq.calls == 0
