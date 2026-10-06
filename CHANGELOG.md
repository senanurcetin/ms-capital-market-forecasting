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
- `make test-order` (`src/evaluation/split_order.py`): `sample_id` is chronological in train at the scale
  of months (block-mean autocorrelation +0.93, -0.05 within a month) and carries no
  order in the test set (+0.01), so features or corrections built from neighbouring samples,
  the regime scaling among them, cannot be computed at scoring time. Results in `results/split_order*`.
- **Learned sequence model** (`src/models/sequence_cnn.py`, `src/models/sequence_experiment.py`,
  `make sequence`): a 176 x 16 snapshot tensor per sample and a small dilated CNN, tested by blending
  it into the tabular model on eight held-out months. Result: the CNN alone scores 0.077
  against 0.123 for the tabular model, and the blend's paired gain is -0.0009 with a
  95% CI of [-0.0104, +0.0086], so no gain was shown. PyTorch is not a dependency
  of CI or the serving image. Results in `results/sequence_probe*`.
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
- **`RidgeModel` fits in chunks.** The full-data `make ship` was killed by the memory limit of a
  16 GB machine inside the Ridge step: `MedianImputer.fit` made a ~4 GB temporary copy, and
  `StandardScaler` plus sklearn's `Ridge` made further float64 copies of the 1.2M x 292 frame.
  Medians are now computed a block of columns at a time, and the scaler and the normal equations
  are accumulated chunk by chunk in float64. On real data (460k training rows) the predictions
  differ from the old pipeline by at most 2.5e-14 and the cosine score is identical, so no
  published number moves; `tests/test_baseline_chunked.py` pins the equivalence. Ridge's peak on
  the full training set fell from 11.7 GB (and then OOM) to 7.2 GB, most of which is the data.

### Dependencies
- GitHub Actions: `actions/checkout` 7, `actions/setup-python` 7, `docker/build-push-action` 7,
  `docker/setup-buildx-action` 4. Dev tools: `pytest` 9.1.1, `mypy` 2.4.0. Dashboard: `plotly` 7.1.0.
  Each was proposed by Dependabot and passed CI on its own; they were applied together because the
  Actions bumps edit adjacent lines of the same workflow files and would conflict one by one.

- `streamlit` 1.64.0, `fastapi` 0.142.2, `uvicorn` 0.54.0, `pydantic` 2.13.5,
  `google-cloud-bigquery` 3.46.1 (+ storage 2.42.0), `db-dtypes` 1.7.2, `duckdb` 1.5.6, `ruff` 0.16.10,
  `pytest-cov` 7.1.0, `docker/login-action` 4, `docker/metadata-action` 6.
- **Security:** `mlflow` 3.15.2 -> 3.16.1 (PYSEC-2026-3865) and a pin of the transitive `cryptography`
  to 50.0.2 (PYSEC-2026-3552). All four requirement sets now pass `pip-audit`; `make audit` covers
  the pipeline set too. The weekly audit would have gone red on both.

### Changed (release)
- `release.yml` can also be started by hand (`workflow_dispatch`) with a version; it then creates the
  annotated tag on the head of `main` itself and publishes both images. It refuses to run from any
  other branch, requires a `vX.Y.Z` version (optionally `-rc.1`), refuses an existing tag that points
  at a different commit, and passes the version to the shell through `env`, not by interpolation.
  Added because a sandboxed session could not push a tag.

### Changed (typing)
- `mypy` now covers the whole project (67 files: `api`, `src`, `streamlit_app`, `scripts`) with no
  exclusions and no ignores. The 30 findings in the training and dashboard code were fixed in place;
  `make demo` was run before and after and its 36 score lines are identical. Almost all were typing
  noise; the one worth a line is that `importance()` on an unfitted model failed with an opaque
  `AttributeError` on `None` and is now a `RuntimeError` saying `fit()` must be called first.

### Fixed
- The MLflow server in `docker-compose.yml` was published on every interface with no authentication;
  it now binds to `127.0.0.1` unless `MSCAPITAL_MLFLOW_BIND` says otherwise.
- Dependabot was opening a single 22-update group that failed as a whole, plus major bumps of the
  numerics stack (pandas 3). The numerics libraries are no longer auto-updated (a bump can change
  predictions or break loading a saved booster); the rest are grouped by what breaks together.
- A non-ASCII `X-Admin-Token` crashed `/reload` with a 500 (`hmac.compare_digest` rejects non-ASCII
  `str`); credentials are now compared as bytes.
- Missing values (NaN) were counted as out-of-range inputs.
