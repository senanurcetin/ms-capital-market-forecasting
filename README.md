# MSCapital — Real Financial Market Forecasting

[![CI](https://github.com/senanurcetin/ms-capital-market-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/senanurcetin/ms-capital-market-forecasting/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)

An end-to-end, production-shaped ML system that predicts short-horizon returns from market
microstructure data — **804.5M raw rows** reduced to a scored model, on a 16 GB laptop.
Built on the Kaggle
[MSCapital](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting)
competition dataset.

`BigQuery` · `LightGBM` · `XGBoost` · `scikit-learn` · `SHAP` · `Optuna` · `Pandera` ·
`MLflow` · `FastAPI` · `Streamlit` · `Docker` · `GitHub Actions`

**Status: complete and handed over.** Five executed notebooks, two scored submissions, and
a test suite that injects the failures it claims to prevent. Nothing is in progress; what
remains open is stated plainly in [Where this stands](#where-this-stands) rather than left
implied.

**Where to start**, depending on how long you have:

| | |
|---|---|
| **2 minutes** | [Headline result](#headline-result) — the scores, and the forecast that was wrong |
| **20 minutes** | ↑ plus [What I found](#what-i-found) and [notebook 01](notebooks/01_data_discovery.ipynb) — the six data findings that drive everything |
| **an hour** | ↑ plus [notebook 05](notebooks/05_why_the_leaderboard_disagreed.ipynb) — six hypotheses, five eliminated, two errors caught in my own analysis |

> **For research only. Not investment advice.**
> The backtesting module exists to measure the model's ranking power, not to propose a strategy.

---

## Headline result

| | cosine | what it is |
|---|---:|---|
| Walk-forward CV mean | **+0.14088** | **the honest internal estimate** — averaged over 5 periods |
| Hold-out, months 65–70 | +0.15171 | measured once, untouched — but a *lucky* period |
| Leaderboard, first submission | +0.12800 | single LightGBM, months 0–63 |
| **Leaderboard, shipped model** | **+0.12900** | ensemble, months 0–67 |

Trained on all 1,257,637 samples and 292 features — though only **264 are actually distinct**; see [Feature redundancy](#feature-redundancy). On the hold-out, Pearson (0.15236) is
almost identical to cosine, confirming predictions are centred on zero — what a
shift-sensitive metric rewards. Directional accuracy 0.5516, RMSE 0.003364.

**The hold-out is listed second on purpose.** It was the original headline: measured once,
on months never touched by feature design, model selection, tuning or early stopping. That
is all true, and it is not the right number. Holding a model fixed and scoring it on every
period in turn shows difficulty swinging from 0.117 to 0.148 — and months 65–70 sit at the
**83rd percentile**. De-biasing for that gives +0.14084; the walk-forward mean, computed a
completely different way, gives +0.14088. Purity was never the binding constraint: a single
contiguous stretch is a sample of size one in the dimension that actually varies.

### A prediction, its falsification, and what followed

Before submitting, notebook 04 recorded a falsifiable forecast — reweighting the hold-out
by the test set's spread mix gave **≈ 0.143**, with the caveat that anything materially
below would need a different explanation.

**Leaderboard: 0.128.** The prediction was wrong: right direction, magnitude off by three.

It stays where it was written, with the correction beneath it, because a forecast is only
evidence of understanding if it is recorded before the answer and reported honestly after.
Five hypotheses were then stated and tested, **none of them costing a further submission**:

| Hypothesis | Verdict |
|---|---|
| **The hold-out was a lucky period** | **confirmed — explains 46%** |
| Spread-regime mix shift | real, but corrected *downward* to ~14% |
| High-drift rate features hurt under shift | falsified at two thresholds |
| Skill decays with elapsed time | falsified — the slope is *positive* |
| Test set is categorically different | falsified — it is a *continuation* of training |

Two results are worth pulling out. The hold-out sits at the **83rd percentile** of period
difficulty (which swings 0.117–0.148 under a fixed model), and de-biasing for that gives
+0.14084 — within 0.00004 of the walk-forward mean, computed a completely different way.
And the forecast itself was **built wrong**: cosine factors exactly as
`Σ cos_g · w_g` with `w_g = ‖y_g‖‖p_g‖ / (‖y‖‖p‖)`, so subgroups are weighted by
*magnitude*, not row count. Correcting it moved the forecast **further** from the outcome.

**→ The full investigation is [notebook 05](notebooks/05_why_the_leaderboard_disagreed.ipynb).**

### Three forecasts, three overshoots

| Prediction | Predicted | Actual |
|---|---:|---:|
| Leaderboard, from the hold-out | 0.143 | 0.128 |
| Spread-mix share of the gap | 26% | ~14% |
| Gain from ensemble + more training data | +0.0047 | +0.0010 |
| Gain from sequence-shape features | clears 0.0041 | +0.0006 |
| Gain from aligning the loss with the metric | small but positive | **−0.0064** |

Different reasoning each time, the same direction of error every time — which points at one
cause rather than three mistakes. **Effects of order 0.002–0.005, measured on internal
splits, sit at this problem's resolution limit.** Fold-to-fold std is 0.0041;
period-to-period std is 0.0091. A CV gap of +0.0022 appearing in 5 folds of 5 is a real
*ordering* of two models and still buys almost nothing externally, because what separates
them is small beside what separates one period from another.

The rule that survives: below roughly the fold-to-fold std, treat an internal gain as
evidence about **which** model to prefer, never as a quantity that will reach a leaderboard.
Both submissions are consistent with that; none of the five forecasts were.

The last one is the sharpest, because it is the only idea that came from this project's own
metric analysis. Cosine weights rows by magnitude, so weighting the training loss by
`|y|^alpha` looked like alignment. It is not: **the weights are a function of the target**,
so the fit no longer estimates `E[y|x]` but a magnitude-tilted functional of it — and cosine
is maximised by the conditional mean. Cosine's weighting describes how scores *aggregate
across rows*, not where a model should spend capacity. Conflating those cost −0.0064 at
alpha 0.5 and −0.0235 at alpha 1.0, with 0 of 6 comparisons improving in either arm.

### Where it actually stands, against other people

187 teams. Median **0.138**. This model **0.129** — rank ~125, below typical. That number
matters because it refutes the comfortable reading of everything above: the problem is
**not** at its noise ceiling. Others reach 0.138 through the same fold-to-fold std of
0.0041, so the missing quantity is signal this pipeline does not extract, not headroom that
does not exist.

Six hypotheses have been tested against that gap. One is confirmed, and it is the
estimator rather than the model: the hold-out sits at the 83rd percentile of period
difficulty, worth ~46% of the shortfall. The other five are eliminated — including the two
most promising, sequence order (**+0.0006**, CI spanning zero, against a pre-registered bar
of 0.0041) and aligning the training loss with the metric (**−0.0064**, which actively
hurt). Each is worked through in
[notebook 05](notebooks/05_why_the_leaderboard_disagreed.ipynb), along with two errors the
audits found in my own analysis.

The honest summary is not "the ceiling is here". It is: **the gap is real, six specific
explanations have been tested, and the one that survives does not account for most of it.**

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
| 6 | The model is **weakest exactly where the test set lives** | the hold-out overstates the leaderboard — measured at 14%, of which this explains ~1/7 | [04](notebooks/04_models_and_errors.ipynb), [05](notebooks/05_why_the_leaderboard_disagreed.ipynb) |

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
| [05 — Why the Leaderboard Disagreed](notebooks/05_why_the_leaderboard_disagreed.ipynb) | **the longest one, and the one to read**: five hypotheses, three falsified, two analysis errors found and corrected |

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

**Code against artefact.** The BigQuery feature tables are *materialised*: generated once,
then queried for months, while the SQL that produced them stays editable in git. The two can
drift apart with no error and no NaN — and once did here, when the generators emitted 82
order and 53 transaction features while BigQuery still held 81 and 52. `make schema-check`
compares the column set the SQL will emit against the column set the tables actually have,
and `tests/test_validation.py` simulates drift in both directions to prove the check fires.

`make validate` runs the contracts against the real data. As with the fold guard,
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
[notebook 05](notebooks/05_why_the_leaderboard_disagreed.ipynb) tests the *mechanism* on training data
alone: train on a fixed window (months 0–34) and evaluate at increasing distance into the
future. If high-drift features were a liability under shift, pruning them would help more
as the gap grows.

It does not. At `|shift| >= 0.2` the trend is flat (r = −0.11); at `|shift| >= 0.1` it is
negative and pruning **costs** skill on average. The hypothesis is falsified at both
thresholds, without a submission.

---

## Hyperparameter search: how much of a gain is real?

Tuning was deprioritised on a judgement. This replaced it with a number.

| | gain |
|---|---:|
| Claimed by the search (23 trials, 30% of rows) | +0.00260 |
| Surviving fresh seeds | +0.00062 |
| **Full data, paired** | **−0.00038** |

**Tuning bought nothing.** Three quarters of the apparent gain was selection noise — keeping
the best of N trials on an objective with fold-to-fold std 0.0041 finds favourable noise as
well as good parameters — and what survived reversed sign on full data. The hand-chosen
defaults stand, now on evidence.

The reusable part is the apparatus: any best-of-N result on a noisy objective is inflated
by the maximum of N noise draws, and the correction costs one extra evaluation. Reporting
only the first number would have claimed a +0.0026 improvement that reverses.

> **A search space is a compute budget in another notation.** The first attempt allowed
> `learning_rate` down to 0.01, where early stopping never fires and every round runs.
> Such trials cost ~10x the baseline and a one-hour search was killed after three with
> nothing to show. The version here bounds the worst trial to ~2x, caps wall-clock, and
> logs every trial as it lands.

---

## Feature redundancy

Auditing a *new* feature set found five of eighteen columns duplicating something that
already existed. The obvious next question was whether the original 292 had the same
problem. They do: **31 pairs correlate above 0.999**, leaving **264 distinct features, not
292**.

| r | pair | why |
|---:|---|---|
| 1.0000 | `txn_intensity_60s` / `txn_n_total` | a rate over a *fixed* window is the count over a constant |
| 1.0000 | `txn_vwap_60s` / `txn_vwap_total` | the widest nested window *is* the whole sample |
| 1.0000 | `mkt_spread_mean_5s` / `mkt_rel_spread_mean_5s` | prices are normalised so mid ≈ 1.0 — dividing by it changes nothing |

The last is the instructive failure: that prices sit near 1.0 was measured and documented
in notebook 01, and the consequence for the feature set was simply never followed up.
Nothing here invalidates a result — boosting is untroubled by correlated inputs — but "292
features" counts columns, not information. `make feature-audit` reproduces it.

---

## Where this stands

**Finished.** The pipeline runs end to end, `make demo` reproduces it on synthetic data in
15 seconds, and every claim above is generated by committed code rather than typed in.

**Open, and stated rather than hidden:**

| | |
|---|---|
| ~54% of the leaderboard gap | unexplained. Six hypotheses tested, five eliminated; the survivor covers 46%. I do not have a seventh. |
| Leaderboard 0.129 vs median 0.138 | below typical. Tuning, ensembling, more data, sequence order and metric alignment are all measured at roughly zero or worse, so what is missing is signal this pipeline does not extract. |
| Prediction horizon | undocumented by the competition; it does not affect the modelling |
| Official metric | confirmed only indirectly — 0.128 is consistent with cosine or Pearson, and inconsistent with RMSE, MAE or R² |

**Five forecasts, five overshoots, all in the same direction.** That pattern is the most
transferable thing here: on this problem, priors about what should help are systematically
optimistic, and only the measurement settles it.

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

## Licence

[MIT](LICENSE) for the code. The competition data is **not** redistributed here and remains
subject to the
[Kaggle competition rules](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting).

Nothing here is investment advice: the backtesting module measures a model's ranking power
for research, and was never evaluated as a trading strategy.

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
