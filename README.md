# MSCapital — Real Financial Market Forecasting

An end-to-end, production-shaped ML system that predicts short-horizon returns from
market microstructure data, built on the Kaggle
[MSCapital](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting)
competition dataset.

> **For research only. Not investment advice.**
> The backtesting module exists to measure the model's ranking power, not to propose a strategy.

---

## Headline result

| | cosine | what it is |
|---|---:|---|
| Walk-forward CV mean | **+0.14088** | **the honest estimate** — averaged over 5 periods |
| Hold-out, months 65–70 | +0.15171 | measured once, untouched — but a *lucky* period |
| **Leaderboard** | **+0.12800** | the only externally graded number |

Trained on all 1,257,637 samples and 292 features. On the hold-out, Pearson (0.15236) is
almost identical to cosine, confirming predictions are centred on zero — what a
shift-sensitive metric rewards. Directional accuracy 0.5516, RMSE 0.003364.

**The hold-out is listed second on purpose.** It was the original headline: measured once,
on months never touched by feature design, model selection, tuning or early stopping. That
is all true, and it is not the right number. Holding a model fixed and scoring it on every
period in turn shows difficulty swinging from 0.117 to 0.148 — and months 65–70 sit at the
**83rd percentile**. De-biasing for that gives +0.14084; the walk-forward mean, computed a
completely different way, gives +0.14088. Purity was never the binding constraint: a single
contiguous stretch is a sample of size one in the dimension that actually varies.

### A prediction, and its falsification

Before submitting, [notebook 04](notebooks/04_models_and_errors.ipynb) recorded a
falsifiable forecast: the model is measurably weakest in tight spreads, the test period
holds more of them, and reweighting the hold-out score by the test spread mix gives
**≈ 0.143** — with the explicit caveat that a score materially below that would need a
different explanation.

**Leaderboard: 0.128.** The prediction was wrong.

| | cosine |
|---|---:|
| Hold-out, as measured | +0.15171 |
| Forecast | +0.14345 |
| **Actual (leaderboard)** | **+0.12800** |

The direction was right and the magnitude wrong by roughly a factor of three. The failed
prediction is left in place with the correction beneath it, because a forecast is only
evidence of understanding if it is recorded before the answer and reported honestly after.

Three follow-ups were then stated and tested. **None of them rescued the story, and no
further submission was spent on any of them:**

| Hypothesis | Method | Verdict |
|---|---|---|
| **The hold-out was a lucky period** | fix the model, score every period in turn | **confirmed — explains 46%** |
| Spread-regime mix shift | reweight hold-out by the test spread mix | real, but **corrected downward** to ~14% |
| High-drift rate features hurt under shift | prune them; vary the train→eval gap over 32 months | **falsified** at two thresholds — pruning is mildly *harmful* |
| Skill decays with elapsed time | same experiment, control arm | **falsified** — slope is *positive* |
| Test set is categorically different | adversarial validation, calibrated against within-train distance | **falsified** — it is a *continuation* |

Two of those need spelling out.

**The test set is not somewhere else.** A classifier separates it from the last training
months at AUC **0.749** — *lower* than months 10–19 against months 0–9 (0.754). Two adjacent
blocks inside the training data are more distinguishable from each other than the test set
is from the end of training. (The first version of this measurement pooled all 71 training
months on one side and got a tidy, wrong answer; the confound is documented, and a test
pins it down.)

**The hold-out was.** With the model held fixed, period difficulty swings from 0.117 to
0.148 and months 65–70 land at the 83rd percentile.

**And the forecast was built wrong.** Cosine is computed over the pooled vector, so it
factors exactly as `cos(y,p) = Σ cos_g · w_g` with `w_g = ‖y_g‖‖p_g‖ / (‖y‖‖p‖)` —
subgroups are weighted by **magnitude, not row count**. The forecast used sample shares.
On the hold-out the true weights run 0.209–0.312 across equal-sized quartiles, and the
heaviest lands on the bucket where the model is *strongest*. Redone properly the forecast
moves to +0.14844 — **further from the outcome**, dropping the spread mechanism's share
from 26% to ~14%.

The two surviving effects should not be added — both say the hold-out period was atypical
and may overlap — so together they account for between **46% and 60%** of a gap that began
fully unexplained. Nor is the remainder noise: the test set averages over ~13 of these
periods, so its own draw is worth ±0.0026 against a residual of 0.0128 — about **5σ**. The
rest is a real effect and still unidentified, which is more useful to report than a tidy
story.
See [notebook 04](notebooks/04_models_and_errors.ipynb).

**What I would do differently:** report the walk-forward mean as the headline and the
hold-out as a *check* on it. A held-out period answers "did I leak?" — not "what will this
score next period". Those were quietly treated as the same question.

> The general lesson outlives this dataset: any subgroup breakdown of a cosine score must
> weight by magnitude or it describes a metric nobody is scored on. A test in
> `tests/test_cosine_decomposition.py` builds a model that is near-perfect on the
> small-magnitude half of the data and useless on the large half — counting rows calls it
> skilful at **+0.48**, while the metric scores it **−0.04**.

The transferable finding is about the *estimator*, not this model: a hold-out carved from
the end of the training period measures generalisation **within one regime**. It reads
optimistically whenever the deployment distribution differs in ways the hold-out cannot
see — here, by 14%.

Walk-forward CV, full data, 5 folds:

| Model | cosine mean | across-fold std |
|---|---:|---:|
| **ensemble** | **+0.14088** | 0.00413 |
| lightgbm | +0.13869 | 0.00332 |
| xgboost | +0.13815 | 0.00471 |
| ridge | +0.11813 | 0.00501 |
| mean | +0.00588 | 0.01374 |
| zero | 0.00000 | — |

The two tree models differ by less than a fifth of the fold-to-fold noise — they are
statistically indistinguishable, so stability decides. The ensemble beats the best single
model in **5 of 5 folds** (median gain +0.0017) using closed-form OLS weights, no tuning.

---

## What I found

Six things here are undocumented, would be got wrong by assumption, and would
each degrade a model *without raising a single error*. Finding them is most of the work in
this repository.

| # | Finding | If missed | Where |
|---|---|---|---|
| 1 | The market table's window is **600 s**, not the 60 s of the other two | 90% of the order-book history is silently discarded | [01](notebooks/01_data_discovery.ipynb) |
| 2 | `price = 0` is an **empty-level sentinel**, not a price | mean relative spread reads **−0.0064** instead of **+0.0013** — the sign flips | [01](notebooks/01_data_discovery.ipynb) |
| 3 | `side` and `order_action` encodings are **recoverable by measurement** | order-flow features get built backwards | [01](notebooks/01_data_discovery.ipynb) |
| 4 | The `*_last` features were reading from **different snapshots** | `mkt_depth_imb1_last` deviated by 1.994 — the full width of its range | [01](notebooks/01_data_discovery.ipynb) |
| 5 | Predictive features and **transferable** features are the same features | — (this one is the payoff, not a trap) | [03](notebooks/03_features_and_drift.ipynb) |
| 6 | The model is **weakest exactly where the test set lives** | the hold-out overstates the leaderboard — measured at 14%, of which this explains ~1/7 | [04](notebooks/04_models_and_errors.ipynb) |

Finding 6 is the one I would lead with in a review, because neither measurement produces it
alone. The drift report says *where the test set sits*: 35.7% of its samples fall in the
tightest-spread quartile, against 25% in training. The error analysis says *where the model
is weak*: 0.1315 cosine in that quartile versus 0.1941 in the widest. Only together do they
say the two overlap — and that turns a vague "performance may vary" into a number that a
leaderboard can falsify. It did: the effect is real but, once the metric's own weighting is
applied, explains only about a seventh of the actual 14% gap. Being specific enough to be
wrong is what made the rest visible.

Two of these were found by checking my own work rather than the data: #4 came from
recomputing features independently instead of re-reading the code that produced them, and
a sixth item — a claim that the 999-row ceiling meant truncation — turned out to be
**over-stated** and is documented as a mistake and its repair, because the lesson
generalises: *an exact round number is evidence of a mechanism, not evidence that the
mechanism matters.*

### Notebooks

Each is executed, with outputs, and generated from a script so the narrative stays
reviewable in version control. All follow the same shape: **what I expected → what I
measured → what I changed.**

| | |
|---|---|
| [01 — Data Discovery](notebooks/01_data_discovery.ipynb) | the four data findings, plus one corrected mistake |
| [02 — The Target, and Why Random Splits Are Banned](notebooks/02_target_and_leakage.ipynb) | the validation rule settled by experiment: a random split inflates the score by +0.0047 (1.04×) |
| [03 — Features, and Whether They Survive the Test Set](notebooks/03_features_and_drift.ipynb) | SHAP × drift: the top-20 features shift 4.2× less than average |
| [04 — Models, and Where They Fail](notebooks/04_models_and_errors.ipynb) | error analysis by liquidity regime — and the risk it exposes |

---

## Run it in 30 seconds

The real pipeline needs Kaggle credentials, a GCP project, ~20 GB of disk and hours of
upload. So there is a second path that needs none of it:

```bash
pip install -r requirements-dev.txt
make demo
```

That generates synthetic data **with the real schema and the real quirks** — one Arrow
record batch per file, a 600 s market window against 60 s elsewhere, `price = 0`
sentinels paired with zero volume, mid-normalised prices — and then drives the genuine
code: the column-group converter, the walk-forward harness with its embargo and runtime
leakage guard, every model, the closed-form ensemble, the hold-out measurement, and
artefact save/reload in the format the API serves. It finishes in about half a minute and
prints the commands to start the API and dashboard on the result.

What it does *not* cover is the BigQuery feature SQL, which needs GCP. That layer is
tested structurally instead (`tests/test_feature_sql.py`), and the demo's feature table
takes its column names from the same SQL generators, so the two schemas cannot drift apart.

It writes to `<repo>/.demo` (override with `MSCAPITAL_DEMO_ROOT`), deliberately **not** to
the configured `data_root` — that points at a large local disk with a platform-specific
absolute path, and using it made the demo Windows-only until CI caught it on Linux.

> The demo's data is synthetic and its signal is planted, so its scores are meaningless
> as results. **No number reported anywhere in this repository comes from it.**

---

## The data: measured facts, not assumptions

Row counts come from the Arrow footers; distributions come from the data itself.

| File | Rows | Cols | On disk | Uncompressed |
|---|---:|---:|---:|---:|
| `train/market.feather` | 221,756,611 | 13 | 4.10 GiB | **11.53 GB** |
| `train/order.feather` | 170,056,583 | 6 | 1.21 GiB | 3.06 GB |
| `train/transaction.feather` | 103,970,264 | 5 | 476 MiB | 1.77 GB |
| `train/label.feather` | 1,257,637 | 3 | 9.6 MiB | — |
| `test/*` (3 files) | 308,733,861 | — | 3.47 GiB | 9.51 GB |
| **Total** | **804,517,319** | | **9.26 GiB** | **25.9 GB** |

**Structure.** Each `sample_id` is an independent, anonymous observation window. There
is **no symbol/instrument column**, so no cross-sample history can be constructed and
the problem reduces to tabular regression over 1,257,637 rows.

**Window lengths differ per table** (measured, not assumed):

| Table | Window | Rows per sample | Note |
|---|---:|---:|---|
| market | **600 s** | 176.3 | one snapshot every ~3.4 s; no ceiling (max 212) |
| order | 60 s | 135.2 | hard ceiling at 999 rows, but it binds for 0.0025% of samples |
| transaction | 60 s | 82.7 | same ceiling, binds for 0.0005% |

The 999 ceiling is real — never exceeded, in either split, for two independent tables —
but measuring *how often it binds* showed it almost never does (31 and 6 samples out of
1,257,637). The `is_truncated` features it originally justified were constant-zero for
99.997% of rows and were removed; `*_window_covered`, which varies continuously, was kept.
See `notebooks/01_data_discovery.ipynb`.

`seconds_before_predict` is the distance back from the prediction instant, sorted
descending within a sample; `0` is the tick closest to prediction time. Because the
value is always `>= 0`, **look-ahead is structurally impossible**.

**Encodings resolved empirically** (prices are normalised around mid ≈ 1.0):

| Code | Meaning | Evidence |
|---|---|---|
| `order.side = 0` | BID | mean price 0.9979 (below mid) |
| `order.side = 1` | ASK | mean price 1.0036 (above mid) |
| `order_action = 0` | NEW | 128.1M events |
| `order_action = 1` | CANCEL | 42.0M events; NEW ≈ CANCEL + TRANSACTION balances |
| `transaction.side = 0` | BUY (aggressor lifts the ask) | 87.5% above mid, mean **+5.26 bps** |
| `transaction.side = 1` | SELL (aggressor hits the bid) | 88.7% below mid, mean **−5.27 bps** |

**`price = 0` is not a price but an "empty level" sentinel** — it always comes with
`volume = 0`. Real prices live in 0.909–1.052. Left uncleaned, mean `rel_spread` reads
−0.0064 instead of the correct +0.001264. There are **no genuinely crossed books** (0 rows).

**Target.** std 0.002618 (26 bps), median exactly 0 (5.54% exact zeros — a tick-size
artefact), autocorrelation between consecutive samples ≈ 0. Monthly std swings by
**2.69×**, i.e. clear regime shift.

---

## Architecture

```
Kaggle feather (single record batch, 11.5 GB uncompressed)
        │  column-group converter (peak RAM 7.9 GB)
        ▼
   Parquet parts ──► BigQuery
        │             mscapital_raw → staging → features → mart
        │             GROUP BY sample_id: 804M rows → 1.26M rows
        ▼
 dataset_train.parquet (1.40 GB, 292 features)
        │
        ▼
 Walk-forward CV ──► MLflow ──► model artefact ──► FastAPI ──► Streamlit
```

### Why a column-group converter

Every competition file is a **single Arrow record batch**. Consequently row-wise
streaming is impossible, `memory_map` is useless (the buffers are compressed), and
`market` cannot be read in one go on a 16 GB machine.

The way out: Arrow IPC compresses each buffer separately and `read_table(columns=[...])`
pushes the projection into the C++ reader (measured: 1 column 0.43 GB, 5 columns
1.15 GB — linear). Market is split into 3 column groups and rejoined in BigQuery on
`row_id`.

That positional-join assumption is verified twice: `tests/test_ingestion.py` performs a
synthetic single-batch round-trip, and in BigQuery the alignment check found **zero**
`sample_id` / `seconds_before_predict` mismatches across 221.7M rows.

---

## The metric: cosine similarity

`cos(y, ŷ) = Σyŷ / (‖y‖·‖ŷ‖)` — **scale-invariant but not shift-invariant.**

- Multiplying predictions by a constant does not change the score → calibrating
  magnitude is wasted effort.
- Adding a bias **hurts**. Empirical proof: the constant-prediction `mean` model scores
  **−0.0036**.
- Ensemble weights need no grid search: the vector in the span of the model predictions
  closest in cosine to `y` is its orthogonal projection, which is the **OLS solution**.
  `tests/test_ensemble.py` verifies this against 200 random weight vectors.

---

## Validation: walk-forward with an embargo

Random splits are **forbidden** — consecutive samples can have overlapping lookback windows.

```
Fold 1: train months 0–34 │ embargo │ val 36–40
...
Fold 5: train months 0–58 │ embargo │ val 60–64
HOLD-OUT (untouchable): months 65–70
```

That ban is measured, not inherited. [Notebook 02](notebooks/02_target_and_leakage.ipynb)
trains the same model on the same data under both splits: the random arrangement scores
**+0.0047 higher (1.04×)**. Real inflation, but far smaller than the rule implies — which
is consistent with the near-zero target autocorrelation. The rule stays, because it costs
nothing and the downside of being wrong is a model that looks good and is not.

`assert_fold_integrity()` re-verifies each fold against the data at runtime.
`tests/test_train_integrity.py` injects broken setups — a random split, an embargo
violation, a hold-out leak — and proves the guard **catches** them.

### Data contracts

Every fact in "measured facts" above is also a Pandera schema in
[`src/data/validation.py`](src/data/validation.py), so a discovery becomes something the
pipeline enforces rather than something a future reader has to rediscover:

| Contract | What it protects |
|---|---|
| `seconds_before_predict >= 0` | look-ahead stays structurally impossible |
| `<= window` (600 s / 60 s / 60 s) | the per-table window semantics |
| `price >= 0`, with 0 allowed | the empty-level sentinel convention |
| `side in {0,1}`, `order_action in {0,1}` | the encodings recovered by measurement |
| `<= 999 rows` per sample | the truncation ceiling |
| descending seconds within a sample | the chronological order `ARRAY_AGG` relies on |

`make validate` runs them against the real data. As with the fold guard,
`tests/test_validation.py` **injects each violation** and asserts it is caught — a schema
nobody has seen fail is a schema nobody knows works.

---

## Explainability

TreeSHAP over the hold-out. The strongest signals are imbalance features, which is what
microstructure theory predicts:

| Feature | Share |
|---|---:|
| `txn_count_imbalance_10s` | 5.48% |
| `ord_new_count_imbalance_30s` | 5.12% |
| `ord_new_count_imbalance_10s` | 3.82% |
| `mkt_micro_minus_mid_last` | 2.83% |
| `mkt_mid_return_60s` | 2.14% |

Contribution by family: market 47.5% · order 32.4% · transaction 20.1%.

### Ablation: SHAP alone answers the wrong question

SHAP is computed on a model that already has every feature, so a family can rank high
simply by encoding information available elsewhere. Retraining on subsets asks the question
that actually matters — what would I lose without it:

| Family | features | SHAP share | alone | **marginal** |
|---|---:|---:|---:|---:|
| `mkt` | 159 | 47.5% | +0.108 | **+0.021** |
| `ord` | 81 | 32.4% | **+0.074** | **+0.017** |
| `txn` | 52 | 20.1% | +0.086 | +0.008 |

The three rankings disagree, and that is the finding. `ord` is the **weakest family on its
own** yet the **second most valuable at the margin**: order flow needs the book to be
interpretable — a burst of new bids means one thing when the spread is wide and another
when it is tight. `txn` runs the other way, respectable alone but largely redundant once
the book and order flow are present, since executed trades are downstream of order flow and
the market table already carries per-snapshot trade aggregates.

All three have positive marginal value, so none is dropped — but that now rests on an
experiment rather than on an importance ranking never designed to answer it.
See [notebook 03](notebooks/03_features_and_drift.ipynb).

### Drift: do the features survive the test set?

Of 292 features, 199 shift negligibly between train and test, 85 slightly, 8 moderately,
and **none** shifts "large" (standardised mean difference ≥ 0.5). But the average hides the
structure:

| Feature kind | median shift |
|---|---:|
| imbalance / OFI / return (scale-free) | **0.002 – 0.007** |
| depth, last-snapshot | 0.021 – 0.028 |
| rate / intensity / count | **0.176** |
| spread | 0.260 |

The test period is a measurably more liquid market — 1.36× order events, 1.35× trade count,
1.24× depth, but **0.80× spread** and 0.91× mid volatility. Everything that *counts* events
moves; everything that measures the *balance* between two counts does not, because a ratio
divides the density out.

That matters because the two rankings converge: the top-20 features by SHAP have a median
shift of **0.0065** versus **0.0275** across all features, and importance correlates
**−0.21** with drift. The features the model leans on are the ones that move least. See
[notebook 03](notebooks/03_features_and_drift.ipynb).

**The honest caveat, and its resolution:** rate features are still in the set and they *do*
shift, so they were the natural suspect for the leaderboard shortfall — and that suspicion
turned out to be wrong. Rather than spend a submission to find out,
[notebook 04](notebooks/04_models_and_errors.ipynb) tests the *mechanism* on training data
alone: train on a fixed window (months 0–34) and evaluate at increasing distance into the
future. If high-drift features were a liability under shift, pruning them would help more
as the gap grows.

It does not. At `|shift| >= 0.2` the trend is flat (r = −0.11); at `|shift| >= 0.1` it is
negative and pruning **costs** skill on average. The hypothesis is falsified at both
thresholds, without a submission.

---

## Hyperparameter search: how much of a gain is real?

Tuning was originally deprioritised on the grounds that it buys little per hour. That was
a judgement; this is the measurement.

By this point the noise floor is known and **high** — fold-to-fold std 0.0041,
block-to-block period std 0.0091. Keeping the best of N trials therefore finds favourable
noise as well as good parameters, so the winner is re-scored under a fresh resampling of
the identical protocol, and then both arms are re-run on full data, paired.

| | gain |
|---|---:|
| Claimed by the search (23 trials, 30% of rows) | +0.00260 |
| Surviving fresh seeds | +0.00062 |
| **Full data, paired, 3 folds** | **−0.00038** |

**Tuning bought nothing.** Three quarters of the apparent gain was selection noise; what
survived was a seventh of the fold-to-fold std; and on full data the tuned configuration
is *worse* than the hand-chosen defaults, improving 1 fold of 3 with per-fold differences
that alternate sign. `min_data_in_leaf` came back at 1018 — tuned against 377k rows, not
1.26M — which is exactly the transfer failure the confirmation step exists to catch.

The reusable part is the apparatus, not the parameters: any best-of-N result on a noisy
objective is inflated by the maximum of N noise draws, and the correction costs one extra
evaluation. Reporting only the first number would have claimed a +0.0026 improvement that
reverses sign on full data.

> **A search space is a compute budget in another notation.** The first attempt allowed
> `num_leaves` 511, `max_bin` 255 and `learning_rate` 0.01 — that last one is the trap,
> since early stopping never fires and every round runs. Trials in that corner cost ~10x
> the baseline, and a search budgeted at an hour was killed after three with nothing to
> show. The version here bounds the worst trial to ~2x, caps wall-clock (it stopped at 23
> of 40 trials and said so), and logs every trial as it lands.

---

## Setup and usage

```bash
pip install -r requirements-dev.txt
make check                 # lint + tests (needs NO live BigQuery and NO downloaded data)
```

`paths.data_root` in `configs/config.yaml` decides where data is written. The default
is `C:/mscapital_data`, deliberately **outside** any synced folder, because the
intermediate data is ~20 GB.

```bash
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

## Cost

BigQuery compresses this data ~5.4× (`market_g2`: 11.57 GiB logical → 2.12 GiB physical).
All four datasets use **physical storage billing**, and `mscapital_raw` is dropped once
staging is verified, leaving ~8 GiB physical — **inside the 10 GiB free tier**. Query
usage sits at roughly 13% of the 1 TiB monthly free allowance. Batch load jobs are free.

---

## Project layout

```
src/
  config.py              single source of paths and constants
  data/                  ingestion (column groups) · bq_loader · staging · mart · test_pipeline
  features/              market (159) · order (81) · transaction (52) · assemble
  evaluation/            metrics (cosine) · temporal_validation · backtesting · explain
  models/                baseline · lightgbm · xgboost · ensemble · train (CLI) · finalize
  inference/             predictor — used by the API, which never imports training code
api/main.py              FastAPI: /health /model-info /predict /batch-predict /reload
streamlit_app/           six-page dashboard
sql/                     BigQuery staging DDL
tests/                   92 tests, none requiring live BigQuery or downloaded data
```
