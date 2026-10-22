# Fixed-fit full-data revalidation — 21 October 2026

The [completed runner job](https://github.com/senanurcetin/ms-capital-market-forecasting/actions/runs/39663832723)
exported the complete existing training feature table using REST `tabledata.list`,
without SQL or the Storage Read API. It verified source metadata identity and every
sample key before writing a local Parquet. No cloud table was created or changed.
Restricted data remained in the ephemeral runner; the temporary repository secret
was removed after completion.

## Protocol and population

- Source: 1,257,637 rows, 292 model features; no sampling.
- Locked settings: [revalidation.json](../configs/revalidation.json).
- LightGBM: 300 rounds, seed 42, two CPU threads; Ridge: alpha 10.
- Five configured expanding-window periods with a one-month embargo.
- Scored validation labels never enter `model.fit`; no outer-fold early stopping.
- Blend weights are fitted only on earlier out-of-training fold predictions.
- Fold 1 supplies calibration history and has no ensemble score.
- Months 65–70 are not trained on or scored by this validation run.

The source table includes the historical holdout rows; the configured fold masks
exclude them. These previously explored historical periods remain retrospective
evidence, not a new independent test set. No parameter search, model promotion,
holdout remeasurement or new competition submission was performed.

## Comparable results

Compare all models on the same four fold IDs, **2–5**:

| Model | Mean fold cosine | Scored periods |
|---|---:|---:|
| LightGBM, fixed 300 rounds | +0.138046 | 4 |
| Ridge, alpha 10 | +0.118520 | 4 |
| Prior-fold calibrated blend | +0.138299 | 4 |

The paired mean blend gain over LightGBM is **+0.000253**. This is a small observed
difference over four historical periods, not proof of a dependable improvement.
The [paired resampling report](../results/walkforward_fixed.uncertainty.json) shows
exploratory circular block-bootstrap intervals and their limitations.
The block-length-1 paired interval is [-0.000118, +0.000469]; block length 2 gives
[+0.000038, +0.000468]. Whether zero is included changes with the block assumption.
Four periods provide too little temporal evidence to treat this as a robust gain.

The raw summary also reports base-model means over all five folds: LightGBM
+0.137479 and Ridge +0.118129. Those five-fold means must not be directly ranked
against the four-fold blend mean. This run uses two members and a fixed tree budget;
its difference from the historical three-member fitted blend does not isolate the
effect of correcting same-fold fitting.

## Immutable run evidence

- [Raw summary](../results/walkforward_fixed.json)
- [Raw summary table](../results/walkforward_fixed.csv)
- [Run record: source/output hashes, code revision, versions, parameters and month counts](../results/walkforward_fixed.run.json)
- [Kaggle publication readback](../results/kaggle_publications.json)

The run used clean commit `eb750b12e9f74a00d633c3aa9ccc45a32cef3731` and completed
at `2026-10-21T20:00:32Z`. The stored Parquet SHA-256 identifies the 1,141,103,601-byte
training input; it does not independently establish its provenance.

The archived runner's `stability.worst_fold`/CSV `worst_fold` is a **one-based position
within the scored rows**, not always an actual fold ID. In the four-fold blend,
position 1 means actual fold 2. The per-fold IDs are authoritative. The harness now
maps this field to actual IDs for future runs; the original hashed output is preserved.

Historical leaderboard and served model artifacts remain unchanged. The Evidence page
displays these new reports separately from the historical study.
