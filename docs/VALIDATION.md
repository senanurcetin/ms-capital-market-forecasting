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
