"""Opt-in, metadata-only integration checks. Never submit a warehouse query or notebook run."""
import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def check_kaggle() -> dict:
    import requests

    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        return {"status": "unavailable", "reason": "KAGGLE_API_TOKEN is not configured"}
    try:
        response = requests.get(
            "https://www.kaggle.com/api/v1/kernels/list",
            params={"user": "senanuretin", "pageSize": "5"},
            headers={"Authorization": f"Bearer {token}"}, timeout=20,
        )
        if response.status_code != 200:
            return {"status": "failed", "http_status": response.status_code}
        rows = response.json()
        if not isinstance(rows, list):
            return {"status": "failed", "reason": "Unexpected metadata response"}
        return {"status": "pass", "scope": "Read-only owner notebook listing", "returned": len(rows)}
    except (requests.RequestException, ValueError) as exc:
        return {"status": "failed", "reason": type(exc).__name__}


def check_bigquery() -> dict:
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account

        key = os.environ.get("MSCAPITAL_GCP_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        credentials = service_account.Credentials.from_service_account_file(key) if key else None
        client = bigquery.Client(
            project=os.environ.get("MSCAPITAL_BQ_PROJECT", "workintech-working"), credentials=credentials,
        )
        datasets = list(client.list_datasets(max_results=20, timeout=20))
        return {"status": "pass", "scope": "Dataset metadata only; no SQL", "returned": len(datasets)}
    except Exception as exc:
        # Report the exception class only: provider messages can contain sensitive paths.
        return {"status": "unavailable", "reason": type(exc).__name__}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--bigquery", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(".artifacts/integrations.json"))
    args = parser.parse_args()
    if not (args.kaggle or args.bigquery):
        parser.error("Choose --kaggle and/or --bigquery to opt into metadata reads")
    checks = {}
    if args.kaggle:
        checks["kaggle"] = check_kaggle()
    if args.bigquery:
        checks["bigquery"] = check_bigquery()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checked_at": datetime.now(UTC).isoformat(), "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks))
    return 0 if all(c["status"] == "pass" for c in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
