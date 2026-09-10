"""Does sequence order carry signal the aggregates do not already have?

The leaderboard is the reason to ask. 204 teams, median 0.137, this model 0.129 at rank
141 - below typical, so the problem is not at its noise ceiling. (standing captured 2026-09-10; `results/leaderboard.json` is what the rest of the project reads) Tuning bought nothing measurable, the
ensemble bought +0.001, more training data bought +0.001. Those levers are spent, so what
is missing is information, and sequence order was the largest identifiable candidate.

A CORRECTION TO THE ORIGINAL PREMISE

This module was first written around the claim that the 292 existing features are all
aggregates, hence permutation-invariant, hence blind to order. That claim was asserted
without checking, and it is wrong: the `*_delta_300s_vs_600s` family compares nested
windows, which is exactly a statement about where a quantity was GOING rather than where
it sat. Order information was already partly present.

The audit below found it. `shp_imb_drift` correlates 0.990 with
`mkt_depth_imb1_delta_300s_vs_600s`, and `shp_n_snaps` correlates 1.000 with
`mkt_snapshot_rate_600s` - a straight duplicate. Five of eighteen exceed 0.9.

THE TEST

Add the 18 shape features ON TOP of the 292 and compare, paired: identical folds,
identical rows, identical seeds, only the column list differing. Shared noise cancels in
the difference, which matters because the fold-to-fold std is 0.0041 and any real effect
here is likely to be of that order.

The decision rule is fixed in advance, so it cannot be bent afterwards:

  gain clears the fold noise        -> sequence structure is real; the sequence-model gate
                                       in the plan opens, and a 1D-CNN/GRU is justified by
                                       measurement rather than by fashion
  gain does not clear it            -> the gate stays shut ON EVIDENCE, which is a result
                                       rather than an omission

A null on its own would not have settled it, though - see audit(), which separates "the
aggregates already have this" from "these features are simply uninformative".
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import cosine_similarity
from src.evaluation.temporal_validation import iter_folds
from src.models.base import feature_columns
from src.models.lightgbm_model import DEFAULT_PARAMS
from src.models.train import assert_fold_integrity, load_dataset

log = logging.getLogger(__name__)

FOLDS = 3
ROUNDS = 800
EARLY_STOPPING = 60


def load_joined() -> tuple[pd.DataFrame, list[str], list[str]]:
    """The feature table with the shape columns joined on.

    Both artefacts are written ordered by sample_id, and the merge is validated one-to-one
    so a silent row duplication cannot inflate the comparison.
    """
    cfg = load_config()
    df = load_dataset("train")
    shape = pd.read_parquet(Path(cfg.paths.features) / "shape_train.parquet")
    for c in shape.columns:
        if c != "sample_id":
            shape[c] = shape[c].astype("float32")

    before = len(df)
    df = df.merge(shape, on="sample_id", how="left", validate="one_to_one")
    assert len(df) == before, "merge changed the row count"

    shape_cols = [c for c in shape.columns if c != "sample_id"]
    base_cols = [c for c in feature_columns(df) if c not in shape_cols]
    log.info("base %d features | shape %d features | %s rows",
             len(base_cols), len(shape_cols), f"{len(df):,}")
    missing = df[shape_cols].isna().mean().mean()
    log.info("shape columns are %.1f%% null on average (LightGBM handles this natively)",
             missing * 100)
    return df, base_cols, shape_cols


def cv(df: pd.DataFrame, cols: list[str], *, seed: int, folds: int = FOLDS) -> list[float]:
    import lightgbm as lgb

    months, y = df["month"].to_numpy(), df["target"].to_numpy()
    params = {**DEFAULT_PARAMS, "seed": seed, "bagging_seed": seed,
              "feature_fraction_seed": seed}

    def cosine_eval(pred, dataset):
        return "cosine", cosine_similarity(dataset.get_label(), pred), True

    scores = []
    for fold, tr, va in list(iter_folds(months))[-folds:]:
        assert_fold_integrity(months, fold, tr, va)
        dtr = lgb.Dataset(df.iloc[tr][cols], label=y[tr])
        dva = lgb.Dataset(df.iloc[va][cols], label=y[va], reference=dtr)
        booster = lgb.train(
            params, dtr, num_boost_round=ROUNDS, valid_sets=[dva], feval=cosine_eval,
            callbacks=[lgb.early_stopping(EARLY_STOPPING, first_metric_only=True,
                                          verbose=False)],
        )
        scores.append(cosine_similarity(y[va], booster.predict(df.iloc[va][cols])))
    return scores


def run(*, seeds: tuple[int, ...] = (0, 1), folds: int = FOLDS) -> dict:
    cfg = load_config()
    df, base_cols, shape_cols = load_joined()
    both = base_cols + shape_cols

    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        b = cv(df, base_cols, seed=seed, folds=folds)
        w = cv(df, both, seed=seed, folds=folds)
        for i, (bs, ws) in enumerate(zip(b, w, strict=False)):
            rows.append({"seed": seed, "fold": i, "base": bs, "with_shape": ws,
                         "diff": ws - bs})
        log.info("seed %d  base %+.5f  with shape %+.5f  diff %+.5f   [%.0f min]",
                 seed, np.mean(b), np.mean(w), np.mean(w) - np.mean(b),
                 (time.perf_counter() - t0) / 60)

    out = pd.DataFrame(rows)
    diffs = out["diff"].to_numpy()
    gain = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    fold_noise = 0.0041           # walk-forward fold-to-fold std, measured in notebook 04

    log.info("\n%s", out.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    log.info("")
    log.info("paired gain      %+.5f  (se %.5f, n=%d)", gain, se, len(diffs))
    log.info("95%% CI          [%+.5f, %+.5f]", gain - 1.96 * se, gain + 1.96 * se)
    log.info("fold noise       %+.5f  <- the bar set in advance", fold_noise)
    log.info("comparisons improved: %d of %d", int((diffs > 0).sum()), len(diffs))
    verdict = "OPEN the sequence-model gate" if gain - 1.96 * se > fold_noise else (
        "gate stays SHUT - measured, not assumed")
    log.info("VERDICT: %s", verdict)

    res = {
        "seeds": list(seeds), "folds": folds, "rounds": ROUNDS,
        "n_base": len(base_cols), "n_shape": len(shape_cols),
        "paired_gain": gain, "se": se, "n_comparisons": len(diffs),
        "ci_low": gain - 1.96 * se, "ci_high": gain + 1.96 * se,
        "fold_noise": fold_noise, "improved": int((diffs > 0).sum()),
        "verdict": verdict,
    }
    dst = Path(cfg.paths.features)
    out.to_csv(dst / "shape_gain.csv", index=False)
    (dst / "shape_gain_meta.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def audit(*, n_sample: int = 200_000, seed: int = 0) -> dict:
    """Is a null result evidence about the hypothesis, or about the features?

    A gain of zero has two explanations that look identical from the outside:

      (a) the features carry information the 292 already have  - the hypothesis is answered
      (b) the features carry no information at all             - the features are the problem

    Reporting (a) without ruling out (b) would be an unsupported claim, so this measures
    both halves: how far each shape feature duplicates an existing one, and what the shape
    features predict on their own.

    Loading only the columns needed - the full 292-column merge costs 1.4 GB and dies on a
    16 GB machine.
    """
    from pathlib import Path

    cfg = load_config()
    lbl = load_dataset("train", columns=["sample_id", "month", "target"])
    shape = pd.read_parquet(Path(cfg.paths.features) / "shape_train.parquet")
    for c in shape.columns:
        if c != "sample_id":
            shape[c] = shape[c].astype("float32")
    shape_cols = [c for c in shape.columns if c != "sample_id"]

    # --- redundancy: nearest existing feature, by absolute correlation ----------------
    base_df = load_dataset("train")
    base_cols = [c for c in feature_columns(base_df) if c not in shape_cols]
    sub = base_df.sample(n=min(n_sample, len(base_df)), random_state=seed)
    sid = sub["sample_id"].to_numpy()
    del base_df

    B = np.nan_to_num(sub[base_cols].to_numpy(dtype=np.float32), nan=0.0,
                      posinf=0.0, neginf=0.0)
    Bz = (B - B.mean(0)) / (B.std(0) + 1e-12)
    sh = shape.set_index("sample_id").loc[sid]
    rows = []
    for c in shape_cols:
        v = np.nan_to_num(sh[c].to_numpy(dtype=np.float32), nan=0.0,
                          posinf=0.0, neginf=0.0)
        vz = (v - v.mean()) / (v.std() + 1e-12)
        r = np.abs(Bz.T @ vz) / len(vz)
        j = int(np.argmax(r))
        rows.append({"shape_feature": c, "max_abs_corr": float(r[j]),
                     "closest_existing": base_cols[j]})
    red = pd.DataFrame(rows).sort_values("max_abs_corr", ascending=False)
    del B, Bz, sub

    # --- standalone power: all of them, and only the genuinely novel ones -------------
    df = lbl.merge(shape, on="sample_id", how="left", validate="one_to_one")
    novel = red.loc[red.max_abs_corr < 0.55, "shape_feature"].tolist()
    all_score = float(np.mean(cv(df, shape_cols, seed=seed, folds=2)))
    novel_score = float(np.mean(cv(df, novel, seed=seed, folds=2)))

    log.info("\n%s", red.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    log.info("duplicates (corr > 0.9): %d of %d",
             int((red.max_abs_corr > 0.9).sum()), len(red))
    log.info("standalone, all %d shape features : %+.5f", len(shape_cols), all_score)
    log.info("standalone, %d genuinely novel    : %+.5f", len(novel), novel_score)

    res = {"n_shape": len(shape_cols), "n_duplicates": int((red.max_abs_corr > 0.9).sum()),
           "n_novel": len(novel), "novel": novel,
           "standalone_all": all_score, "standalone_novel": novel_score,
           "median_max_corr": float(red.max_abs_corr.median())}
    dst = Path(cfg.paths.features)
    red.to_csv(dst / "shape_redundancy.csv", index=False)
    (dst / "shape_audit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Does sequence shape add signal?")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--audit", action="store_true",
                    help="skip the gain test; check redundancy and standalone power")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.audit:
        audit()
    else:
        run(seeds=tuple(args.seeds), folds=args.folds)


if __name__ == "__main__":
    main()
