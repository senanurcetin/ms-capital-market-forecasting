"""Resumable table exports never run SQL and reject source drift or key corruption."""
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from src.config import Config
from src.data.export_rest import export_features


class RestTable:
    def __init__(self):
        self.table = SimpleNamespace(full_table_id="p:d.dataset_train", etag="fixed", num_rows=6,
                                     num_bytes=96, modified=datetime(2026, 10, 21, tzinfo=UTC))
        self.calls = []
        self.keys = list(range(6))
        self.drift = False
        self.lookups = 0

    def get_table(self, name, timeout):
        self.lookups += 1
        if self.drift and self.lookups > 1:
            return SimpleNamespace(**{**vars(self.table), "etag": "changed"})
        return self.table

    def list_rows(self, table, *, start_index, max_results, page_size, timeout):
        self.calls.append(start_index)
        def arrow(*, create_bqstorage_client):
            assert create_bqstorage_client is False
            return pa.table({"sample_id": self.keys[start_index:start_index + max_results],
                             "signal": [1.] * max_results})
        return SimpleNamespace(to_arrow=arrow)


@pytest.fixture
def config():
    cfg = Config({"samples": {"train": 6, "test": 4},
                  "bigquery": {"project": "p", "datasets": {"features": "d"}}})
    with patch("src.data.export_rest.load_config", return_value=cfg):
        yield


def test_export_and_resume_use_rest_only_and_preserve_all_keys(tmp_path, config):
    bq = RestTable()
    output = export_features(tmp_path, bq=bq, chunk_rows=2, workers=1)
    assert pq.read_table(output).column("sample_id").to_pylist() == list(range(6))
    source = json.loads(output.with_suffix(".source.json").read_text())
    assert source["query_jobs"] == 0 and source["unique_keys"] == 6
    bq.calls.clear()
    export_features(tmp_path, bq=bq, chunk_rows=2, workers=1)
    assert bq.calls == []


def test_source_change_during_export_prevents_final_publication(tmp_path, config):
    bq = RestTable()
    bq.drift = True
    with pytest.raises(RuntimeError, match="changed"):
        export_features(tmp_path, bq=bq, chunk_rows=2, workers=1)
    assert not (tmp_path / "dataset_train.parquet").exists()


def test_changed_source_cannot_reuse_partial_files(tmp_path, config):
    bq = RestTable()
    export_features(tmp_path, bq=bq, chunk_rows=2, workers=1)
    bq.table.etag = "different"
    with pytest.raises(ValueError, match="Source or chunk size changed"):
        export_features(tmp_path, bq=bq, chunk_rows=2, workers=1)


@pytest.mark.parametrize("keys", [[0, 1, 2, 3, 4, 4], [0, 1, 2, 3, 4, 7], [0, 1, 2, 3, 4, None]])
def test_invalid_sample_keys_cannot_publish(tmp_path, config, keys):
    bq = RestTable()
    bq.keys = keys
    with pytest.raises(ValueError, match="sample keys|expected sample key"):
        export_features(tmp_path, bq=bq, chunk_rows=2, workers=1)
    assert not (tmp_path / "dataset_train.parquet").exists()
