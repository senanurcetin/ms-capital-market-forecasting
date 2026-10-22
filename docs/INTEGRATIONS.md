# Kaggle, BigQuery and zero-cost operation

[Documentation index](README.md)

## Data lineage

```text
Kaggle competition Feather files (restricted, stored outside Git)
  → Arrow projections → chunked Parquet column groups
  → BigQuery raw → staging (row alignment / no-look-ahead checks)
  → feature blocks → one row per sample → local Parquet
  → walk-forward training, untouched holdout, separate submitted ensemble
  → derived results/ exports → Streamlit + static HTML report
```

Configuration is in `configs/config.yaml`; the default historical project is
`workintech-working`, location `US`, with `mscapital_raw`, `mscapital_staging`,
`mscapital_features` and `mscapital_mart` datasets. These names document the pipeline;
their presence in source is not proof that remote tables still exist.

| Integration | Recorded source | Verification boundary |
|---|---|---|
| Competition | [MSCapital](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting) | Download requires the owner's access and acceptance of competition rules |
| Main public notebook | [My Hold-out Was a Lucky Stretch](https://www.kaggle.com/code/senanuretin/ms-capital-my-hold-out-was-a-lucky-stretch) | Remote notebook is separate from local executed notebooks |
| Follow-up notebook | [Four Experiments That Did Not Help](https://www.kaggle.com/code/senanuretin/ms-capital-four-experiments-that-did-not-help) | Not automatically republished by this repository |
| Leaderboard | `results/leaderboard.json` | Historical capture dated 2026-09-10, not current standing |
| Warehouse | `configs/config.yaml`, `src/data/`, `src/features/`, `sql/` | Local tests do not certify current cloud credentials, IAM, storage or billing |
| Predictions | `results/model_meta.json`, `results/model.txt` | Bundled LightGBM v3; submitted ensemble v4 is distinct |

## Credentials and project isolation

Use `MSCAPITAL_DATA_ROOT` for data outside the checkout, `MSCAPITAL_BQ_PROJECT` for your
own project and `MSCAPITAL_GCP_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` for a key path.
Kaggle credentials remain in the user's Kaggle configuration. Never paste key contents
into notebooks, issues, reports or environment examples. Viewing the app needs no keys.

### Connection check on 19 October 2026

The local Kaggle API credential was rejected with HTTP 401 when listing the owner's
notebooks. Neither remote notebook status nor a fresh Kaggle run was verified.
No Application Default Credentials or configured repository service-account key was
available for BigQuery, so no warehouse query or write was submitted. The historical
exports remain usable offline. Restoring live access requires valid owner credentials;
the committed integration code and public notebook links alone cannot establish it.

### Recovery on 20 October 2026

The replacement owner-provided token successfully authenticated a read-only Kaggle
owner notebook listing (HTTP 200, five returned records). The GitHub repository secret
was updated through secret storage; no token is committed. This verifies metadata access,
not notebook execution, data download or a new competition submission.

The subsequently supplied service-account file authenticated BigQuery metadata reads
and two dry runs in `workintech-working` (US). The
[dated warehouse snapshot](../results/warehouse_audit.json) records 22 tables across
the four configured datasets, nine passing row-count/schema contracts, and all 292
bundled model features present as numeric columns in both assembled datasets.
Train has 1,257,637 rows and test has 647,896 rows. Train/test feature names and types
match after excluding train-only `month` and `target`; test contains neither.
These are metadata checks, not fresh uniqueness/null-rate scans or a reproduced score.

`mscapital_raw` exists but is empty, consistent with the documented raw-layer removal.
The other layers total 74,506,418,814 logical bytes (about 69.4 GiB), above the
10 GiB storage allowance. Logical bytes are not a billing statement: storage model,
account allowance, retention and billing details still require an account review.
This pass neither deleted data nor created new tables.

Dry runs estimate 2,772,746,896 bytes for the ordered training download and
1,432,128,368 bytes for test. Both exceed the default 1,000,000,000-byte job ceiling.
No data scan, feature rebuild, download, training, table mutation or Kaggle submission
was executed. Raising the ceiling is a separate decision after reviewing remaining
free allowance; a successful schema audit does not authorize a cost-bearing rebuild.

To repeat the scoped audit with a key path in `MSCAPITAL_GCP_KEY`:

```powershell
python -m src.data.warehouse_audit --dry-run --output .artifacts/warehouse.json
```

Without `--dry-run`, the audit reads table metadata only. The command exits nonzero
for missing required tables, row/type mismatches, train/test schema drift, leaked
train-only columns, unavailable metadata or failed dry runs. Provider error details
and credential contents are excluded. The Evidence page displays the bundled dated
snapshot offline and never opens a cloud connection.

The opt-in `Integration metadata` GitHub Actions workflow can check the existing
repository `KAGGLE_API_TOKEN` without exposing or copying its value to a workstation.
It lists owner notebook metadata only: no notebook execution, submission or data download.
Locally, use `python scripts/check_integrations.py --kaggle` with an environment token,
or `--bigquery` for ADC/service-account dataset metadata. Missing credentials fail visibly.

### Notebook updates on 21 October 2026

Both existing public Kaggle notebooks were updated with the same-fold ensemble
correction and completed execution on Kaggle CPU: the main holdout analysis at version
4 and the experiment summary at version 2. Their source and metadata are versioned in
[kaggle/](../kaggle/README.md). They re-execute the historical analysis and plot;
they do not retrain the competition pipeline or submit predictions. The opt-in
`scripts/check_kaggle_publications.py` checks the latest completed run and compares
ordered cell-source hashes with the versioned copies.

The manual `Research revalidation` workflow separately supports a full-data run of
the locked protocol in `configs/revalidation.json`. It uses a temporary repository
secret `MSCAPITAL_REVALIDATION_GCP_KEY`, materializes it only under the runner's
temporary directory, and deletes the file in an always-run cleanup step. Restricted
feature data stays in that ephemeral runner; only derived metrics and hashed run
records are uploaded. Remove the temporary repository secret after the run.
This workflow is opt-in, never runs on pull requests and does not publish or submit
Kaggle notebooks. The default dispatch performs notebook readback only.

The [21 October full-data run](REVALIDATION.md) completed successfully. It performed
REST data reads, not SQL queries, and published only derived reports. The temporary
repository secret was removed after completion. The [Kaggle readback snapshot](../results/kaggle_publications.json)
also verifies completed execution and matching sources for both public notebooks.

## Cost boundaries

The portfolio path is the committed results bundle, local compute and a static report;
it needs no database, paid API, custom domain or payment integration. There is no
monetization flow. The project demonstrates research and engineering capability.

Google documents a free allowance of 10 GiB storage and 1 TiB on-demand queries per
month. Verify current regional/account terms in the
[official free-tier documentation](https://docs.cloud.google.com/free/docs/free-cloud-features#bigquery)
before running a rebuild. This dataset's raw + staging + feature layers can exceed
storage allowances. Partitioned scans still count, and repeated runs share the same
account allowance. Historical full-scale processing is not a promise of a free rebuild.

The central client now applies `maximum_bytes_billed=1_000_000_000` to each query by
default, including the shape-feature path. Set `MSCAPITAL_BQ_MAX_BYTES` to a positive
integer only after reviewing the scan. Invalid or unlimited values fail before client
creation. A query above the ceiling fails; it is not automatically retried without it.
An explicitly injected third-party BigQuery client must enforce its own budget.

A per-job ceiling cannot enforce a monthly $0 budget or constrain storage, network,
load jobs or separately configured clients. Use a sandbox/account quota, dry-run SQL
first, and check remaining allowance before any cloud job. Budget alerts alone are not
a hard spending cap. See Google's [cost control guide](https://docs.cloud.google.com/bigquery/docs/best-practices-costs).
The audit uses Google's [dry-run query mode](https://docs.cloud.google.com/bigquery/docs/samples/bigquery-query-dry-run)
to estimate downloads without running them. It has no fallback to query execution.

Keep training offline after downloading the compact feature table. Avoid scheduled
full refreshes, retain only intentionally needed cloud layers, and review storage
before creating replacements. The current development pass does not delete remote data.

## Scale triggers

| Trigger | Response within the zero-cost constraint |
|---|---|
| Repeated warehouse scans | Reuse local Parquet; inspect dry-run bytes before another query |
| Cloud storage approaches its allowance | Stop ingestion; inventory retention before rebuilding |
| Streamlit memory or cold-start pressure | Share static `site/index.html`; keep training outside the app |
| Export bundle approaches 24 MB | Keep aggregate reports, bound samples, preserve model/schema compatibility |
| New result or model promotion | Re-export, audit, rebuild site, check model identity and rerun relevant tests |
