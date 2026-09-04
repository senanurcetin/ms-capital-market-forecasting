"""Is the test set LATER than training, or DIFFERENT from it?

The leaderboard came in 14% under the hold-out, and notebook 04 rejected the two obvious
explanations: pruning high-drift features does not help, and skill does not decay with
elapsed time anywhere in the 71-month training span. That leaves the possibility that the
test set is not a continuation of the training period at all, but something categorically
apart from it.

"Categorically apart" is testable. Train a classifier to tell train rows from test rows:
the AUC measures how distinguishable the two distributions are. But an AUC on its own says
nothing - features drift with time, so ANY two separated periods are somewhat
distinguishable. The number needs a scale.

THE CALIBRATION

So the same measurement is repeated *inside* the training data, at increasing temporal
distance: months 0-9 against 10-19, against 20-29, and so on. That traces out how
distinguishability grows with elapsed time, in this market, for these features. The
train-vs-test AUC can then be read against that curve rather than against intuition.

  If the test AUC sits ON the curve  -> the test set is simply "later", and the
                                        degradation should have been foreseeable
  If it sits far ABOVE the curve     -> no amount of elapsed time explains it; the test
                                        set differs in kind, and extrapolating from the
                                        training period was never going to work

Both arms use identical model settings and equal class sizes, so the AUCs are comparable.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.base import NON_FEATURES

log = logging.getLogger(__name__)

REFERENCE = (0, 9)
TARGETS = [(10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 70)]
PER_SIDE = 40_000
PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 5.0, "max_bin": 127, "verbosity": -1, "num_threads": 0,
}
ROUNDS = 200


def _auc(a: pd.DataFrame, b: pd.DataFrame, features: list[str], seed: int = 0) -> tuple:
    """AUC separating two samples, plus the features that drive it.

    Fit and score on disjoint halves so the AUC is out-of-sample: an in-sample AUC would
    measure memorisation, not distinguishability.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    X = pd.concat([a[features], b[features]], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    idx = rng.permutation(len(X))
    cut = len(X) // 2
    tr, va = idx[:cut], idx[cut:]

    booster = lgb.train({**PARAMS, "seed": seed},
                        lgb.Dataset(X.iloc[tr], label=y[tr]), num_boost_round=ROUNDS)
    auc = float(roc_auc_score(y[va], booster.predict(X.iloc[va])))
    imp = pd.Series(booster.feature_importance("gain"), index=features)
    return auc, imp.sort_values(ascending=False)


def _load_test_sample(path, features: list[str], n: int, seed: int = 0) -> pd.DataFrame:
    """Sample the test feature table without materialising all of it.

    Row groups are drawn evenly across the file so the sample spans the whole test set;
    reading the first few groups would take a contiguous run of sample_id instead.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    groups = list(range(pf.num_row_groups))
    per_group = max(1, pf.metadata.num_rows // max(1, pf.num_row_groups))
    want = max(1, min(len(groups), int(np.ceil(n * 2 / per_group))))
    chosen = np.linspace(0, len(groups) - 1, want).round().astype(int)
    tbl = pf.read_row_groups(sorted(set(chosen.tolist())),
                             columns=["sample_id"] + features)
    df = tbl.to_pandas()
    del tbl
    for c in features:                       # match the training dtype exactly
        df[c] = df[c].astype("float32")
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def _sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=seed) if len(df) > n else df


def run(*, per_side: int = PER_SIDE, seed: int = 0) -> pd.DataFrame:
    from src.models.train import load_dataset

    cfg = load_config()
    train = load_dataset("train")
    features = [c for c in train.columns if c not in NON_FEATURES]
    # Only `per_side` test rows are ever needed, and materialising all 648k x 292 as
    # pandas costs 1.5 GB. Read row groups spread evenly across the file instead - taking
    # the first few would sample a contiguous stretch of sample_id, which is not the same
    # thing as sampling the test set.
    test = _load_test_sample(Path(cfg.paths.features) / "dataset_test.parquet",
                             features, per_side, seed)
    log.info("train %s rows | test sample %s rows | %d features",
             f"{len(train):,}", f"{len(test):,}", len(features))

    ref = _sample(train[train.month.between(*REFERENCE)], per_side, seed)
    rows, importances = [], {}

    for lo, hi in TARGETS:
        other = _sample(train[train.month.between(lo, hi)], per_side, seed)
        auc, imp = _auc(ref, other, features, seed)
        # distance between block midpoints, in months
        dist = ((lo + hi) / 2) - ((REFERENCE[0] + REFERENCE[1]) / 2)
        rows.append({"comparison": f"months {lo}-{hi}", "kind": "within-train",
                     "distance_months": dist, "auc": auc, "n_per_side": len(other)})
        importances[f"{lo}-{hi}"] = imp
        log.info("  ref vs months %2d-%2d  (distance %4.1f)  AUC %.4f", lo, hi, dist, auc)

    # The comparisons of interest. These must be BLOCK vs BLOCK, exactly like the rows
    # above: pooling all 71 training months makes that side far more heterogeneous, which
    # depresses the AUC for a reason that has nothing to do with the test set. Comparing a
    # pooled train sample against the curve would understate the shift.
    test_s = test
    auc_t, imp_t = _auc(ref, test_s, features, seed)
    rows.append({"comparison": "TEST (vs reference block)", "kind": "train-vs-test",
                 "distance_months": np.nan, "auc": auc_t, "n_per_side": per_side})
    importances["TEST"] = imp_t
    log.info("  ref (months %d-%d) vs TEST              AUC %.4f", *REFERENCE, auc_t)

    last = _sample(train[train.month.between(*TARGETS[-1])], per_side, seed)
    auc_last, imp_last = _auc(last, test_s, features, seed)
    rows.append({"comparison": "TEST (vs last block)", "kind": "train-vs-test",
                 "distance_months": np.nan, "auc": auc_last, "n_per_side": per_side})
    importances["TEST_vs_last"] = imp_last
    log.info("  months %d-%d vs TEST                    AUC %.4f", *TARGETS[-1], auc_last)

    # And the pooled version, kept only to show how much the confound matters.
    auc_pool, _ = _auc(_sample(train, per_side, seed), test_s, features, seed)
    rows.append({"comparison": "TEST (vs pooled train)", "kind": "diagnostic",
                 "distance_months": np.nan, "auc": auc_pool, "n_per_side": per_side})
    log.info("  pooled train vs TEST                   AUC %.4f  (not comparable)", auc_pool)

    out = pd.DataFrame(rows)
    within = out[out.kind == "within-train"]
    slope, intercept = np.polyfit(within.distance_months, within.auc, 1)
    # Where the curve would put a test set 36 months past the end of training.
    implied_distance = (auc_t - intercept) / slope if slope else np.nan

    log.info("\n%s", out.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    log.info("within-train trend: AUC = %.5f + %.6f * months", intercept, slope)
    log.info("max within-train AUC %.4f; TEST vs reference %.4f; TEST vs last block %.4f",
             within.auc.max(), auc_t, auc_last)
    log.info("to reach the TEST AUC by elapsed time alone would take %.0f months "
             "(training spans 71)", implied_distance)

    dst = Path(cfg.paths.features)
    out.to_csv(dst / "adversarial_auc.csv", index=False)
    top = pd.DataFrame({k: v.head(20) for k, v in importances.items()})
    top.to_csv(dst / "adversarial_top_features.csv")
    (dst / "adversarial_meta.json").write_text(json.dumps({
        "per_side": per_side, "rounds": ROUNDS, "reference": list(REFERENCE),
        "test_auc_vs_reference": auc_t, "test_auc_vs_last_block": auc_last,
        "test_auc_vs_pooled_train": auc_pool,
        "max_within_train_auc": float(within.auc.max()),
        "within_slope_per_month": float(slope), "within_intercept": float(intercept),
        "implied_distance_months": float(implied_distance),
    }, indent=2), encoding="utf-8")

    log.info("\ntop discriminators, train vs test:\n%s",
             imp_t.head(12).to_string(float_format=lambda v: f"{v:,.0f}"))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Adversarial validation with a temporal scale")
    ap.add_argument("--per-side", type=int, default=PER_SIDE)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(per_side=args.per_side)


if __name__ == "__main__":
    main()
