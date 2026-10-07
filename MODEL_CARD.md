# Model card — MSCapital short-horizon return model

> **For research only. Not investment advice.** Nothing here was evaluated as a trading
> strategy; the backtest module measures ranking power, not profitability.

Every figure below is read from [`results/`](results/) and checked against it by
`tests/test_model_card.py`, so this page cannot drift from the pipeline unnoticed.

## Model

| | |
|---|---|
| Task | regress a short-horizon return from order-book, order and transaction features |
| Served by default | LightGBM, version `v3` (`results/model.txt`) |
| Submitted to the leaderboard | LightGBM (months 0-63) and a LightGBM + XGBoost + Ridge blend (months 0-67), the blend scoring higher |
| Inputs | 292 engineered features, of which 264 are distinct (`results/feature_audit.json`) |
| Output | a predicted return; the API also reports UP / DOWN / FLAT, which is presentation only |
| Metric | cosine similarity between predictions and targets, pooled over the whole test set |

**Which model the numbers belong to.** The hold-out measurements describe the LightGBM,
because it was trained on months 0-63 and so months 65-70 are untouched for it. The blend
trains through month 67 and therefore has no hold-out score; its only external measurement
is its leaderboard result. The README explains the choice.

## Intended use

Studying how much short-horizon signal market-microstructure features carry, how that signal
holds up across time, and how an internal estimate relates to an external score. It is a
worked, audited example of that investigation.

**Not for:** placing trades, sizing positions, or any decision with money attached. The
model has never been run live, its horizon is undocumented by the competition, and the
backtest ignores everything a real execution would face.

## Data

The Kaggle [MSCapital](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting)
competition data: 804.5M raw rows across three tables, reduced to 1,257,637 labelled training
samples over 71 months. The competition data is **not** redistributed here; `results/` holds
derived aggregates and a 4,970-row feature sample drawn evenly across all months.

## Performance

| | cosine | note |
|---|---:|---|
| Walk-forward CV, ensemble, mean of 5 folds | **+0.14088** | the honest internal estimate; fold-to-fold std 0.00413 |
| Walk-forward CV, LightGBM | +0.13869 | std 0.00332 |
| Hold-out, months 65-70, LightGBM | +0.15171 | measured once, but a lucky period (83rd percentile of difficulty) |
| **Leaderboard, shipped ensemble** | **+0.129** | rank 141 of 204; median 0.137, best 0.172 (captured 2026-09-10) |

On the hold-out: Pearson 0.15236, directional accuracy 0.5516, RMSE 0.003364.

The internal estimate and the leaderboard differ by about 0.012. Roughly 46% of the gap to
the hold-out is the period-difficulty effect, about 14% is a shift in the spread regime, and
the rest is unexplained — see [notebook 05](notebooks/05_why_the_leaderboard_disagreed.ipynb).

## Limitations

- **Below the median.** The shipped model scores under the leaderboard median. The missing
  quantity is signal the pipeline does not extract, not headroom that does not exist.
- **Effects of order 0.002-0.005 are below this problem's resolution.** Fold-to-fold std is
  0.0041 and period-to-period std 0.0091. Five forecasts of internal gains overshot in the
  same direction; treat a gain smaller than the fold std as evidence about *which* model to
  prefer, never as a number that will reach a leaderboard.
- **Weakest where the test set lives.** Error is highest in the tightest-spread regime, which
  holds 35.7% of the test samples against 25% of training.
- **Hand-built sequence statistics add nothing** (+0.0006, CI spanning zero). That ruled out
  those 18 statistics, not the idea that order carries signal. A learned sequence model (a small
  CNN over the 600 s market window) was tried afterwards: it scores **0.077** on its own, but
  blending it into the tabular model was not shown to help (paired gain **-0.0009**, 95% CI
  **[-0.0104, +0.0086]** over eight held-out months), so the idea is neither confirmed nor
  ruled out at that width. Neither a regime estimated from the feature space nor four further
  ensemble members helped either; the shipped model is unchanged (`results/sequence_probe.json`,
  `regime_clusters.json`, `ensemble_probe.json`).
- **Metric confirmed only indirectly.** A score of 0.128 is consistent with cosine or Pearson
  and inconsistent with RMSE, MAE or R².
- **Static.** Trained once on a fixed period. No retraining schedule or live drift response
  exists; the API only counts out-of-range inputs, and only for artefacts that carry
  `feature_ranges.json`.

## Reproducing and checking

`make demo` runs the whole pipeline on synthetic data in about 15 seconds. `make check`
runs lint, type checks and the tests, none of which need credentials or the competition data.

## Links

- Code, tests and the full write-up: [GitHub](https://github.com/senanurcetin/ms-capital-market-forecasting),
  release [v1.1.0](https://github.com/senanurcetin/ms-capital-market-forecasting/releases/tag/v1.1.0)
- Model files and this card on [Hugging Face](https://huggingface.co/senanurcetin/ms-capital-market-forecasting)
- [Live dashboard](https://ms-capital-market-forecasting-mfy6rngulq4fpaovzrhntf.streamlit.app/)
- Kaggle notebooks: [My Hold-out Was a Lucky Stretch](https://www.kaggle.com/code/senanuretin/ms-capital-my-hold-out-was-a-lucky-stretch)
  and [Four Experiments That Did Not Help](https://www.kaggle.com/code/senanuretin/ms-capital-four-experiments-that-did-not-help)
