# Verification record — 19 October 2026

## Follow-up — 20 October 2026

Historical ensemble CV is now explicitly identified as a same-fold fitted-blend
diagnostic. New training calibrates only on earlier folds, preserving actual fold IDs.
Paired block-bootstrap comparisons exclude the historical blend and align model periods.
The short README links to the preserved full narrative in `docs/RESEARCH.md`.
CLI training now emits a hashed, credential-free experiment sidecar. Regression checks
cover frozen-weight target independence, actual fold identity and sidecar creation.
The supplied replacement Kaggle token verified an owner metadata listing. The later
owner-supplied BigQuery key verified 22 tables, nine metadata contracts, and schema
compatibility for all 292 served model features. Two download dry runs completed;
both exceed the default scan ceiling and neither was executed. The dated
[warehouse report](../results/warehouse_audit.json) records scope, estimates and storage
warnings. Metadata counts do not validate uniqueness or reproduce historical scores.
No historical experiment result artifact was overwritten.

The preceding change also passed all four GitHub Actions jobs, including both Docker
builds and a real synthetic artifact served through the API image. That remote check
supersedes the Docker-unavailable boundary in the original 19 October table below.
New warehouse regression tests cover missing tables, row/type drift, test target
leakage, model feature compatibility, sanitized failures, dry-run-only execution,
schema fingerprints and explicit query billing-project selection.

[Documentation index](README.md)

This record separates local verification from remote integration and deployment status.
The historical research metrics and model artifacts were retained.

| Check | Result |
|---|---|
| Complete offline pytest suite | Passed; four optional-dependency tests skipped; coverage 72.02%, above the configured 68% floor |
| Ruff | Passed across application, pipeline, tests and scripts |
| Mypy | Passed across 80 source files |
| Streamlit AppTest | Router and all eight research pages rendered, including local prediction and evidence |
| Synthetic end-to-end pipeline | 2,500 samples: ingestion, 292-feature assembly, five embargoed folds, holdout, model save/reload and prediction completed |
| Evidence reconciliation | 48 bundled files inventoried; CSV/JSON/fold headline metrics agree |
| Documentation links | All local link targets in the documentation set resolve |
| Official logo provenance | Transparent wordmark SHA-256 matches the source file in the owner's Drive Logo folder |
| Docker build | Not executed: Docker Desktop Linux engine was unavailable |
| Live Kaggle / BigQuery | Not verified; see [connection status](INTEGRATIONS.md#connection-check-on-19-october-2026) |
| Public Streamlit deployment | The running public revision has not been certified by these local checks |

The synthetic demo's planted signal is not a competition measurement. No new Kaggle
submission, paid cloud job, warehouse mutation or model promotion was performed.
The revised theme uses native surfaces for System/light/dark preferences so custom
colors cannot force dark backgrounds under native light navigation text. Final checks
use Streamlit AppTest and deterministic tests; no Playwright is required.

The Python validation environment lives outside the OneDrive checkout to avoid syncing
installed dependencies. Temporary reports are ignored under `.artifacts/`; data and
credentials stay outside the tracked project. Historical screenshot evidence retains
its original date and is not presented as a capture of this revision.
