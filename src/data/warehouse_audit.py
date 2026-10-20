"""Opt-in warehouse metadata contracts and dry-run estimates; never execute SQL scans.

Run: python -m src.data.warehouse_audit --dry-run --output .artifacts/warehouse.json
Metadata row counts do not prove key uniqueness, null rates or model performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT, Config, load_config
from src.data.bq_loader import client
from src.data.query_budget import query_config


def _schema(table: Any) -> dict[str, str]:
    return {field.name: field.field_type for field in table.schema}


def _contracts(cfg: Config) -> dict[tuple[str, str], tuple[int, dict[str, str]]]:
    contracts = {("staging", "label"): (cfg.samples.train, {
        "sample_id": "INTEGER", "month": "INTEGER", "target": "FLOAT",
    })}
    for split in ("train", "test"):
        for name in ("market", "order", "transaction"):
            required = {"sample_id": "INTEGER", "seconds_before_predict": "FLOAT"}
            if split == "train":
                required["month"] = "INTEGER"
            contracts[("staging", f"{name}_{split}")] = (cfg.expected_rows[split][name], required)
        required = {"sample_id": "INTEGER"}
        if split == "train":
            required.update(month="INTEGER", target="FLOAT")
        contracts[("features", f"dataset_{split}")] = (cfg.samples[split], required)
    return contracts


def audit_warehouse(bq: Any, *, cfg: Config | None = None,
                    model_features: list[str], dry_run: bool = False) -> dict[str, Any]:
    """Inspect only configured datasets. Provider error messages/credentials are not exported."""
    cfg = cfg or load_config()
    errors: list[str] = []
    warnings: list[str] = []
    datasets: list[dict[str, Any]] = []
    found: dict[tuple[str, str], Any] = {}
    for layer, name in cfg.bigquery.datasets.items():
        entry: dict[str, Any] = {"layer": layer, "dataset": name, "tables": []}
        datasets.append(entry)
        try:
            dataset = bq.get_dataset(f"{cfg.bigquery.project}.{name}", timeout=20)
            entry["location"] = dataset.location
            if dataset.location != cfg.bigquery.location:
                errors.append(f"{name}: unexpected location {dataset.location}")
            for item in bq.list_tables(dataset, timeout=20):
                table = bq.get_table(item.reference, timeout=20)
                schema = [field.to_api_repr() for field in table.schema]
                entry["tables"].append({
                    "table": table.table_id, "rows": table.num_rows,
                    "logical_bytes": table.num_bytes, "columns": len(table.schema),
                    "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                    "modified": table.modified.isoformat() if table.modified else None,
                })
                found[(layer, table.table_id)] = table
            entry["tables"].sort(key=lambda t: t["table"])
            entry["status"] = "pass"
            if not entry["tables"]:
                warnings.append(f"{name}: empty dataset; raw ingestion cannot be replayed from cloud raw tables")
        except Exception as exc:
            entry.update(status="unavailable", reason=type(exc).__name__)
            errors.append(f"{name}: metadata unavailable ({type(exc).__name__})")

    checks = []
    for (layer, name), (expected_rows, required) in _contracts(cfg).items():
        table = found.get((layer, name))
        issues = []
        if table is None:
            issues.append("missing table")
        else:
            if table.num_rows != expected_rows:
                issues.append(f"rows {table.num_rows}; expected {expected_rows}")
            actual = _schema(table)
            for column, kind in required.items():
                if actual.get(column) != kind:
                    issues.append(f"{column}: expected {kind}, found {actual.get(column)}")
            if layer == "features":
                if name.endswith("_test"):
                    leaked = sorted({"target", "month"}.intersection(actual))
                    if leaked:
                        issues.append(f"unexpected test metadata: {leaked}")
                for feature in model_features:
                    if actual.get(feature) not in ("FLOAT", "INTEGER", "NUMERIC", "BIGNUMERIC"):
                        issues.append(f"model feature absent or nonnumeric: {feature}")
                if not model_features:
                    issues.append("empty model feature contract")
        checks.append({"table": f"{layer}.{name}", "expected_rows": expected_rows,
                       "status": "fail" if issues else "pass", "issues": issues})
        errors.extend(f"{layer}.{name}: {issue}" for issue in issues)

    train, test = found.get(("features", "dataset_train")), found.get(("features", "dataset_test"))
    if train is not None and test is not None:
        train_schema = {k: v for k, v in _schema(train).items() if k not in {"target", "month"}}
        if train_schema != _schema(test):
            errors.append("Assembled train/test feature names or types differ")

    logical_bytes = sum(t.num_bytes or 0 for t in found.values())
    if logical_bytes > 10 * 1024**3:
        warnings.append("Logical table bytes exceed 10 GiB; inspect account storage billing. This is not a bill estimate.")

    plans = []
    if dry_run:
        for split in ("train", "test"):
            if ("features", f"dataset_{split}") not in found:
                continue
            # Fixed read-only SQL; no arbitrary SQL, writes, result fetch or fallback execution.
            sql = (f"SELECT * FROM `{cfg.bigquery.project}.{cfg.bigquery.datasets.features}.dataset_{split}` "
                   "ORDER BY sample_id")
            config = query_config()
            config.dry_run = True
            config.use_query_cache = False
            try:
                job = bq.query(sql, job_config=config, location=cfg.bigquery.location, timeout=20)
                estimate = job.total_bytes_processed
                plans.append({"split": split, "sql": sql, "estimated_bytes": estimate,
                              "maximum_bytes_billed": config.maximum_bytes_billed,
                              "within_job_ceiling": estimate <= config.maximum_bytes_billed if estimate is not None else None,
                              "executed": False, "status": "pass"})
            except Exception as exc:
                plans.append({"split": split, "executed": False, "status": "unavailable", "reason": type(exc).__name__})
                errors.append(f"{split}: dry run unavailable ({type(exc).__name__})")

    return {
        "checked_at": datetime.now(UTC).isoformat(), "project": cfg.bigquery.project,
        "status": "fail" if errors else "pass", "scope": "Metadata contracts and optional dry runs only",
        "data_queries_executed": 0, "model_feature_count": len(model_features),
        "table_count": len(found), "logical_bytes": logical_bytes,
        "datasets": datasets, "contracts": checks, "download_plans": plans,
        "errors": errors, "warnings": warnings,
        "limitations": ["Metadata counts do not establish key uniqueness, null rates or target validity.",
                        "Schema compatibility does not prove export identity or reproduce historical scores.",
                        "A per-job ceiling does not establish remaining monthly free allowance."],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Estimate both feature downloads without executing SQL")
    parser.add_argument("--output", type=Path, default=Path(".artifacts/warehouse.json"))
    args = parser.parse_args()
    meta = json.loads((REPO_ROOT / "results/model_meta.json").read_text(encoding="utf-8"))
    try:
        report = audit_warehouse(client(), model_features=meta["features"], dry_run=args.dry_run)
    except Exception as exc:
        report = {"status": "unavailable", "reason": type(exc).__name__,
                  "checked_at": datetime.now(UTC).isoformat(), "data_queries_executed": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "checked_at", "data_queries_executed")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
