# Changelog

Notable changes to the code, the serving API and the published results. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Before this file existed, the history is
`git log`; the leaderboard standing and every score are read from `results/`, not from here.

## [Unreleased]

### Added
- **Model registry** (`src/inference/registry.py`, `make promote / rollback / releases`): every
  promotion is kept as an immutable release, `current` is swapped for the new one, and a release
  that has been rolled away from is not returned to by a later rollback.
- **Provenance** in `model_meta.json`: git revision and dirty flag, a hash of the ordered feature
  list, training rows and months. Reported by `/v1/model-info`.
- **`MSCAPITAL_API_KEY`**: when set, the model endpoints need an `X-API-Key` header. Open when unset.
- `scripts/e2e_smoke.py` / `make smoke`, and a CI job that serves the demo artefact from the API
  image and drives it over HTTP.
- `MSCAPITAL_BQ_PROJECT` to point the pipeline at a BigQuery project other than the author's.
- `MODEL_CARD.md`, checked against `results/` by a test; `make prune` (feature-pruning experiment,
  not yet run on the full data); `GET /v1/features`; `/v1` routes; `/metrics`; request-size and
  rate limits; `feature_ranges.json` and an out-of-range input counter.
- `CONTRIBUTING.md`, `SECURITY.md`, issue and pull-request templates, Dependabot, a weekly
  `pip-audit` workflow, pre-commit, mypy on the clean modules, and a tag-triggered image release.

### Changed
- **`POST /reload` is disabled until `MSCAPITAL_ADMIN_TOKEN` is set** (403), and then needs a
  matching `X-Admin-Token` (401 otherwise). Deployments that call it must set the token.
- `MSCAPITAL_DATA_ROOT` relocates every data path, not only the API's.
- The coverage floor rose from 60% to 68% (measured 71%).

### Fixed
- A non-ASCII `X-Admin-Token` crashed `/reload` with a 500 (`hmac.compare_digest` rejects non-ASCII
  `str`); credentials are now compared as bytes.
- Missing values (NaN) were counted as out-of-range inputs.
