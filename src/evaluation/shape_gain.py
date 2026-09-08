"""Does sequence order carry signal the aggregates do not already have?

The 292 existing features are all aggregates, and aggregates are permutation-invariant:
shuffle the ~176 snapshots inside a sample and not one of them changes. Whatever lives in
the ORDER of the book's evolution is therefore absent from the model by construction.

The leaderboard says something is absent. 187 teams, median 0.138, this model 0.129.
Tuning bought nothing measurable, the ensemble bought +0.001, and more training data
bought +0.001 - so the missing quantity is information, not method. Sequence order is the
largest identifiable candidate, and this measures whether it pays.

THE TEST

Add the 18 shape features ON TOP of the 292 and compare, paired: identical folds,
identical rows, identical seeds, only the column list differing. Shared noise cancels in
the difference, which matters because the fold-to-fold std is 0.0041 and any real effect
here is likely to be of that order.

The decision rule is fixed in advance, so it cannot be bent afterwards:

  gain clears the fold noise        -> sequence structure is real; the sequence-model gate
                                       in the plan opens, and a 1D-CNN/GRU is justified by
                                       measurement rather than by fashion
  gain does not clear it            -> the aggregates already capture what order provides;
                                       the gate stays shut ON EVIDENCE, which is a result
                                       rather than an omission

Both outcomes are worth having. The second is the one that would otherwise be an
unexamined hole in the project.
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
        for i, (bs, ws) in enumerate(zip(b, w)):
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Does sequence shape add signal?")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--folds", type=int, default=FOLDS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(seeds=tuple(args.seeds), folds=args.folds)


if __name__ == "__main__":
    main()
