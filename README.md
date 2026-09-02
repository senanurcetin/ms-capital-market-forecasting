# MSCapital — Real Financial Market Forecasting

An end-to-end, production-shaped ML system that predicts short-horizon returns from
market microstructure data, built on the Kaggle
[MSCapital](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting)
competition dataset.

> **For research only. Not investment advice.**
> The backtesting module exists to measure the model's ranking power, not to propose a strategy.

---

## Headline result

Measured **once** on the hold-out months (65–70, 36,669 samples), which were never
touched during feature design, model selection or tuning:

| Metric | Value |
|---|---:|
| **Cosine similarity** | **0.14853** |
| Pearson correlation | 0.14946 |
| Directional accuracy | 0.5484 |
| RMSE | 0.003354 |

The hold-out score matches the walk-forward CV estimate (0.144), so the model is not
overfitted to the validation folds. Pearson being almost identical to cosine confirms
predictions are centred on zero — exactly what a shift-sensitive metric rewards.

| Model (walk-forward CV) | cosine mean | std | dir. acc |
|---|---:|---:|---:|
| **ensemble** | **+0.1455** | 0.0084 | — |
| lightgbm | +0.1440 | 0.0088 | 0.550 |
| xgboost | +0.1398 | 0.0053 | 0.549 |
| ridge | +0.1241 | 0.0100 | 0.543 |
| zero | 0.0000 | — | — |
| mean | −0.0036 | 0.0010 | 0.522 |

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
| market | **600 s** | 176.3 | one snapshot every ~3.4 s; no row cap (max 212) |
| order | 60 s | 135.2 | **caps at 999 rows** → truncation feature |
| transaction | 60 s | 82.7 | **caps at 999 rows** → truncation feature |

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
 dataset_train.parquet (1.39 GB, 294 features)
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

Random splits are **forbidden** — consecutive samples can have overlapping windows.

```
Fold 1: train months 0–34 │ embargo │ val 36–40
...
Fold 5: train months 0–58 │ embargo │ val 60–64
HOLD-OUT (untouchable): months 65–70
```

`assert_fold_integrity()` re-verifies each fold against the data at runtime.
`tests/test_train_integrity.py` injects broken setups — a random split, an embargo
violation, a hold-out leak — and proves the guard **catches** them.

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

Contribution by family: market 41.6% · order 32.7% · transaction 25.7% — all three
source tables earn their place.

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
make features    # BigQuery feature layer + local download
make train       # walk-forward + MLflow
make api         # FastAPI   :8000
make streamlit   # Dashboard :8501
```

With Docker:

```bash
docker compose up -d      # api :8000, streamlit :8501, mlflow :5000
```

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
  features/              market (159) · order (82) · transaction (53) · assemble
  evaluation/            metrics (cosine) · temporal_validation · backtesting · explain
  models/                baseline · lightgbm · xgboost · ensemble · train (CLI) · finalize
  inference/             predictor — used by the API, which never imports training code
api/main.py              FastAPI: /health /model-info /predict /batch-predict /reload
streamlit_app/           six-page dashboard
sql/                     BigQuery staging DDL
tests/                   92 tests, none requiring live BigQuery or downloaded data
```
