# Verification record — 19 October 2026

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
