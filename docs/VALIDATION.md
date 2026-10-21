# Validation protocol and uncertainty

The historical `results/` bundle is retained unchanged. Review on 20 October 2026 found
that its ensemble weights were fitted and evaluated on the same fold targets. The
recorded +0.14088 is therefore a fitted-blend diagnostic; agreement with a period-bias
correction does not establish independence or unbiased performance. The separately
measured LightGBM holdout and Kaggle submission scores are not recomputed by this change.

New `run_walk_forward` ensemble evaluation fits nonnegative weights on earlier folds'
out-of-training predictions and freezes them before scoring the current fold. Fold 1
has no earlier calibration data, so it has no ensemble score. Selecting the last N folds
preserves their real fold IDs. Current targets cannot change the fitted blend weights.
Base-model early stopping still uses its validation fold; fully nested outer validation
would require a separate full-data experiment and is not claimed here.

`python -m src.evaluation.revalidate` is a separate, opt-in retrospective check with
the fixed settings in `configs/revalidation.json`: LightGBM at 300 rounds and Ridge
at alpha 10, no sampling, no early stopping on the scored periods, and blend calibration
on earlier folds only. `run_walk_forward(..., fit_validation=False)` passes no scored
validation data to `model.fit`. Months 65–70 are never scored again by this command.
Previously explored historical folds remain retrospective evidence even with this
cleaner fitting boundary; they do not become a new independent test set.

The command requires the complete local training Parquet and writes a separate summary
and hashed run record. It never replaces the historical summary or promotes a model.
If the table is absent, `python -m src.data.export_rest --destination <outside-Git-path>`
can resume a bounded REST `tabledata.list` export without SQL or the Storage Read API.
It checks source identity, total rows and every expected sample key before publishing
the local file. Streaming part assembly avoids holding the full float64 table in RAM.

`python -m src.evaluation.uncertainty` writes `site/uncertainty.json`. It:

- aligns all eligible models on the same consecutive fold IDs;
- excludes the historical same-fold ensemble unless its protocol explicitly records prior-fold calibration;
- reports mean fold cosine and paired deltas relative to LightGBM;
- resamples shared circular block indices with lengths 1 and 2, seed 42, 10,000 draws;
- records the source summary SHA-256 and assumptions.

The percentile 95% resampling intervals are exploratory. Five periods provide little
information about temporal dependence; expanding training sets overlap, regimes may
change, and circular blocks wrap the final period to the first. Length 1 is only an IID
sensitivity check. These are not pooled-row cosine intervals, significance tests, or
calibrated confidence bounds on future competition performance.

For paired resampling principles see [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
For circular temporal blocks see [CMU's time-series bootstrap notes](https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ADAfaEPoV.pdf).

CLI training writes a sibling `.run.json`: source Parquet SHA-256, summary SHA-256,
selected row/month counts, feature-order hash, sampling seed/fraction, validation config,
model parameters, code commit/dirty state and package versions. Input hashing is streamed
and repeated after training; changed inputs prevent publication. Hashes identify bytes,
not independent provenance. Historic artifacts do not acquire retroactive run records.
