"""Metadata contracts fail visibly without ever scanning data or exporting secrets."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.cloud import bigquery
from src.config import load_config
from src.data.bq_loader import client
from src.data.warehouse_audit import audit_warehouse


class Warehouse:
    def __init__(self):
        cfg = load_config()
        self.tables = {}
        self.queries = []
        for split in ("train", "test"):
            for name in ("market", "order", "transaction"):
                schema = [bigquery.SchemaField("sample_id", "INTEGER"),
                          bigquery.SchemaField("seconds_before_predict", "FLOAT")]
                if split == "train":
                    schema.append(bigquery.SchemaField("month", "INTEGER"))
                self.add("staging", f"{name}_{split}", cfg.expected_rows[split][name], schema)
            schema = [bigquery.SchemaField("sample_id", "INTEGER"), bigquery.SchemaField("signal", "FLOAT")]
            if split == "train":
                schema.extend([bigquery.SchemaField("month", "INTEGER"), bigquery.SchemaField("target", "FLOAT")])
            self.add("features", f"dataset_{split}", cfg.samples[split], schema)
        self.add("staging", "label", cfg.samples.train, [bigquery.SchemaField("sample_id", "INTEGER"),
                 bigquery.SchemaField("month", "INTEGER"), bigquery.SchemaField("target", "FLOAT")])

    def add(self, layer, name, rows, schema):
        dataset = load_config().bigquery.datasets[layer]
        self.tables[f"{dataset}.{name}"] = SimpleNamespace(
            table_id=name, num_rows=rows, num_bytes=100, schema=schema, modified=None,
        )

    def get_dataset(self, ref, timeout):
        return SimpleNamespace(dataset_id=ref.split(".")[-1], location="US")

    def list_tables(self, dataset, timeout):
        return [SimpleNamespace(reference=k) for k in self.tables if k.startswith(dataset.dataset_id + ".")]

    def get_table(self, ref, timeout):
        return self.tables[ref]

    def query(self, sql, *, job_config, location, timeout):
        assert job_config.dry_run is True
        assert job_config.use_query_cache is False
        assert sql.startswith("SELECT * FROM `")
        self.queries.append(sql)
        # Deliberately no result(): the audit must never fetch query results.
        return SimpleNamespace(total_bytes_processed=2_000_000_000)


def test_contracts_and_dry_runs_never_execute_or_fetch_data():
    warehouse = Warehouse()
    report = audit_warehouse(warehouse, model_features=["signal"], dry_run=True)
    assert report["status"] == "pass", report["errors"]
    assert len(report["contracts"]) == 9
    assert report["table_count"] == 9
    assert len(warehouse.queries) == 2
    assert all(plan["executed"] is False for plan in report["download_plans"])
    assert all(plan["within_job_ceiling"] is False for plan in report["download_plans"])
    assert report["data_queries_executed"] == 0


def test_metadata_only_is_the_default():
    warehouse = Warehouse()
    assert audit_warehouse(warehouse, model_features=["signal"])["status"] == "pass"
    assert warehouse.queries == []


@pytest.mark.parametrize("fault", ["missing", "rows", "type", "target_leak", "feature_missing", "empty_contract"])
def test_broken_warehouse_contract_is_not_a_success(fault):
    warehouse = Warehouse()
    table = warehouse.tables["mscapital_features.dataset_test"]
    features = ["signal"]
    if fault == "missing":
        del warehouse.tables["mscapital_features.dataset_test"]
    elif fault == "rows":
        table.num_rows -= 1
    elif fault == "type":
        table.schema[1] = bigquery.SchemaField("signal", "STRING")
    elif fault == "target_leak":
        table.schema.append(bigquery.SchemaField("target", "FLOAT"))
    elif fault == "feature_missing":
        features.append("new_signal")
    else:
        features = []
    report = audit_warehouse(warehouse, model_features=features)
    assert report["status"] == "fail"
    assert report["errors"]


def test_train_test_extra_feature_drift_is_rejected():
    warehouse = Warehouse()
    warehouse.tables["mscapital_features.dataset_train"].schema.append(bigquery.SchemaField("extra", "FLOAT"))
    assert "Assembled train/test feature names or types differ" in audit_warehouse(
        warehouse, model_features=["signal"],
    )["errors"]


def test_provider_failures_are_sanitized_and_fail_closed():
    warehouse = Warehouse()
    with patch.object(warehouse, "get_dataset", side_effect=PermissionError("secret-value private-key-path")):
        report = audit_warehouse(warehouse, model_features=["signal"])
    assert report["status"] == "fail"
    assert "secret-value" not in json.dumps(report)
    assert "PermissionError" in json.dumps(report)


def test_dry_run_failure_cannot_fall_back_to_execution():
    warehouse = Warehouse()
    with patch.object(warehouse, "query", side_effect=PermissionError("private-key-path")) as query:
        report = audit_warehouse(warehouse, model_features=["signal"], dry_run=True)
    assert report["status"] == "fail"
    assert query.call_count == 2
    assert "private-key-path" not in json.dumps(report)
    assert all(plan["executed"] is False for plan in report["download_plans"])


def test_schema_hash_changes_when_column_type_changes():
    warehouse = Warehouse()
    before = audit_warehouse(warehouse, model_features=["signal"])
    warehouse.tables["mscapital_features.dataset_test"].schema[1] = bigquery.SchemaField("signal", "INTEGER")
    after = audit_warehouse(warehouse, model_features=["signal"])
    assert before["datasets"] != after["datasets"]


def test_storage_is_a_warning_separate_from_contract_success():
    warehouse = Warehouse()
    warehouse.tables["mscapital_features.dataset_test"].num_bytes = 11 * 1024**3
    report = audit_warehouse(warehouse, model_features=["signal"])
    assert report["status"] == "pass"
    assert any("exceed 10 GiB" in warning for warning in report["warnings"])


def test_project_override_controls_job_billing_project():
    cfg = copy.deepcopy(load_config())
    cfg["bigquery"]["project"] = "isolated-project"
    with patch("src.data.bq_loader.load_config", return_value=cfg), patch(
        "src.data.bq_loader.bigquery.Client.from_service_account_json",
    ) as constructor:
        client()
    assert constructor.call_args.kwargs["project"] == "isolated-project"
    assert constructor.call_args.kwargs["default_query_job_config"].maximum_bytes_billed > 0


def test_committed_live_snapshot_has_explicit_scope():
    path = Path(__file__).resolve().parents[1] / "results/warehouse_audit.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["table_count"] == 22
    assert report["model_feature_count"] == 292
    assert report["data_queries_executed"] == 0
    assert len(report["contracts"]) == 9
    assert len(report["download_plans"]) == 2
