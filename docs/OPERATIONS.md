# Operations and experiment commands

[Documentation index](README.md) · [Quick start](REPRODUCIBILITY.md)

## Model releases

A model used to be whatever sat in `models/current`, so replacing it overwrote the previous one.
`src/inference/registry.py` keeps every promotion:

```bash
make promote DIR=<model dir>   # validate by loading it, store a copy as a release, serve it
make releases                  # newest first, * marks the served one
make rollback                  # serve the previous release; a release rolled away from is skipped
python -m src.inference.registry activate <release>   # a deliberate jump to any release
```

then `POST /reload` on the API. Releases are copies, not symlinks (this is developed on Windows,
where a symlink needs elevated rights); promotion refuses an artefact the API could not load; if
the swap fails the previous `current` is restored. Each artefact also records its provenance in
`model_meta.json` - git revision (and whether the tree was dirty), a hash of the *ordered*
feature list, training rows and months - which `/model-info` reports.

## Setup and usage

```bash
pip install -r requirements-dev.txt
make check                 # lint + typecheck + tests (needs NO live BigQuery and NO downloaded data)
make audit                 # pip-audit the pinned runtime requirement sets
pre-commit install         # optional: ruff + mypy + file hygiene on every commit
make deploy-check          # renders every dashboard page on the runtime deps ALONE
```

`deploy-check` is separate from `check` for a reason worth stating, because it was learned
the expensive way. `make check` runs in whatever virtualenv you are in, and a development
machine always has more installed than the deployment does. A call to
`Styler.background_gradient` reaches for matplotlib inside pandas at render time -
no import statement to catch, matplotlib present locally from the pipeline work, 235 tests
green - and every page on Streamlit Community Cloud raised `ImportError`. `deploy-check`
builds an empty interpreter, installs `requirements.txt` and nothing else, and renders
each page. **A clean-clone test verifies the repository; it says nothing about the
environment.** Those are two different checks and both are needed.

`paths.data_root` in `configs/config.yaml` decides where data is written. The default
is `C:/mscapital_data`, deliberately **outside** any synced folder, because the
intermediate data is ~20 GB. Set `MSCAPITAL_DATA_ROOT` to relocate every data path at once
- the pipeline, the API, the dashboard, the Makefile and docker compose all read it.

The BigQuery project in `configs/config.yaml` is the author's own, which nobody else can bill or
read: set `MSCAPITAL_BQ_PROJECT` to run the pipeline against yours (the datasets are created
inside it by `make ingest`).

`POST /reload` swaps the served model, so it is **off by default**: it answers 403 until
`MSCAPITAL_ADMIN_TOKEN` is set, and then needs a matching `X-Admin-Token` header.

`MSCAPITAL_API_KEY` is the credential for everything else: **open when unset** (a local run
and the compose stack stay zero-config), and once set `/model-info`, `/features`, `/predict` and
`/batch-predict` need a matching `X-API-Key` header - the dashboard sends it from the same
variable. `/health` and `/metrics` stay open so orchestrators and scrapers need no credential,
and `/reload` keeps its own admin token so an operator does not need both. Unauthenticated
requests still pass through the rate limiter, so guessing a key is throttled.

The API also guards what one request may cost, all through environment variables:

| Variable | Default | Effect |
|---|---|---|
| `MSCAPITAL_MAX_BATCH_ROWS` | 10000 | rows per `/batch-predict` (422 above it) |
| `MSCAPITAL_MAX_BODY_BYTES` | 20 MiB | 413 before the body is parsed; chunked uploads without a length get 411 |
| `MSCAPITAL_RATE_LIMIT_PER_MIN` | 0 (off) | per-client limit on `/predict` and `/batch-predict`, 429 with `Retry-After` |

The rate limit is per process and keyed on the client address, so it is off by default:
behind a reverse proxy every client shares one address, and a shared limit belongs in the
proxy. `GET /metrics` serves Prometheus text - request counts by route template and status,
latency sums, and rows scored. It carries counts only, never request contents, and unknown
paths share one `unmatched` label so a scanner cannot inflate it.

The model endpoints (`/model-info`, `/predict`, `/batch-predict`, `/reload`) are served under
`/v1`, which is the documented contract, and still answer at the root for existing clients
such as the dashboard. `/health` and `/metrics` are operational and unversioned.

Using it - `/v1/features` lists the inputs, and every one is required (gaps are not filled in):

```bash
curl -s localhost:8000/v1/features | python -c "import sys,json; print(json.load(sys.stdin)['n_features'])"
python - <<'PY'
import requests
base = "http://localhost:8000/v1"
names = requests.get(f"{base}/features").json()["features"]
row = dict.fromkeys(names, 0.0)                      # replace with real feature values
print(requests.post(f"{base}/predict", json={"features": row}).json())
PY
```

When the artefact carries `feature_ranges.json` (written by `ship.py`, `finalize.py` and the
demo from the 0.1%-99.9% band of the training features), the API counts values outside that
band per feature in `/metrics` as `mscapital_out_of_range_values_total`. It is a drift signal
only: the row is still scored. Artefacts built before this simply skip the check.

```bash
make pipeline    # all of the steps below, in order (hours; needs the data and BigQuery)
make ingest      # feather → parquet → BigQuery → staging
make validate    # data contracts (Pandera) on raw + features
make features    # BigQuery feature layer + local download
make train       # walk-forward + MLflow
make api         # FastAPI   :8000
make streamlit   # Dashboard :8501
```

With Docker:

```bash
docker compose up -d      # api :8000, streamlit :8501, mlflow :5000
```

The build is multi-stage, one target per service, because the single-image version was
**3.48 GB**: the API was inheriting MLflow, Streamlit, SHAP and its numba/llvmlite stack,
DuckDB, Polars, the Kaggle client and the GCP clients — none of which are needed to load
an artefact and score a row.

**Which artefact is published, and why the dashboard shows two models' numbers.** Both
models were submitted and both were graded: the single LightGBM on months 0-63 scored
**0.128**, and the ship.py ensemble on months 0-67 scored **0.129**. The ensemble is the
better model and its score is the one quoted as the leaderboard result.

What travels in `results/` is nonetheless the LightGBM. The reason is the hold-out: it was
trained on months 0-63, so months 65-70 are untouched for it and the 0.15171 cosine, the
backtest curve and the SHAP importances are all measurable. The ensemble trains through
month 67 - deliberately, because a hold-out has done its job the moment it is read - which
buys four more months of data and forfeits any hold-out score at all. Publishing it would
leave the Predictions page serving a model with no measured performance beside it.

So the page carries the LightGBM's measurements and the ensemble's leaderboard result, and
says which is which rather than implying one model produced both.

That constraint shapes how the ensemble is stored. `ship.py` fits three models and
generates the submission with their blend, so the blend is what produced the leaderboard
score — but the servable artefact used to be the bare LightGBM booster saved under the
name `ensemble`, so `/model-info` announced a blend while `/predict` returned one model's
predictions. Pickling the fitted wrappers would have been one line and would have dragged
`src.models` into the API image, undoing the separation above. Instead each base model is
written in a format the runtime dependencies can already read — booster files for LightGBM
and XGBoost, and for Ridge the fitted sklearn objects **plus the imputation medians**,
because unwrapping it once lost those and `predict()` then raised `Input X contains NaN`
on the first gap in the test set. `predict_test()` loads that same artefact, so the
submission and the endpoint cannot drift apart.

| Image | Size | Contents |
|---|---:|---|
| `mscapital:api` | **777 MB** | serving deps + `src/inference` only |
| `mscapital:app` | 2.49 GB | + Streamlit and read-only BigQuery |
| `mscapital:full` | 3.48 GB | everything, for training and MLflow |

Two measurements drove most of the 78% reduction on the API image. The default `xgboost`
wheel pulls the **NVIDIA CUDA runtime — 454 MB inside the image** — which is dead weight
for CPU-only serving, so it uses `xgboost-cpu`. And `pyarrow` (153 MB) went too: the
predictor never reads Parquet, it takes JSON rows.

That the API target can be built from `src/config.py` and `src/inference/` alone is also
the cleanest proof that it does not depend on the training code — a static import check
confirms `predictor` never reaches into `src.models` or `src.evaluation`.

### Credentials

- **Kaggle**: `~/.kaggle/kaggle.json`
- **GCP**: `MSCAPITAL_GCP_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` (these override
  `configs/config.yaml`). Keep the key file outside the repository.

---
