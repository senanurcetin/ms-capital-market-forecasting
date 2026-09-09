"""Train the loss the metric actually rewards.

Every model in this project is fitted with L2 and then SCORED with cosine. That mismatch
was noted early and never acted on, and the cosine decomposition makes the size of it
concrete: a pooled cosine is

    cos(y, p) = SUM_g cos_g * w_g,   w_g = ||y_g|| ||p_g|| / (||y|| ||p||)

so the metric weights samples by MAGNITUDE. A sample whose target is 3 sigma from zero
counts for far more than one sitting at the tick, and 5.5% of targets are exactly zero and
contribute nothing at all. L2 does not know any of that: it weights every row identically
and spends capacity fitting rows the metric ignores.

THE INTERVENTION

Weight each training row by |y|^alpha. alpha = 0 is the current behaviour; alpha = 1
matches the linear magnitude weighting the metric applies; alpha = 0.5 is the compromise,
included because the full weighting throws away most of the data's influence and could
easily overshoot.

This is the one remaining idea in the project that comes out of its OWN metric analysis
rather than from general practice, and it has never been tried.

WHAT WOULD COUNT AS SUCCESS

The same bar as everywhere else: the paired gain has to clear the fold-to-fold noise of
0.0041. Given that tuning bought nothing, the ensemble bought +0.001 and sequence shape
bought nothing, the prior is not good - but unlike those, this one changes what the model
is optimising rather than how it is fitted, so it is not obviously in the same class.

The comparison is paired: identical folds, rows and seeds, only the weights differing.
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

ALPHAS = (0.0, 0.5, 1.0)
FOLDS = 3
ROUNDS = 800
EARLY_STOPPING = 60


def weights(y: np.ndarray, alpha: float) -> np.ndarray | None:
    """Row weights proportional to |y|^alpha, normalised to mean 1.

    Returns None for alpha = 0 so the unweighted arm takes exactly the current code path
    rather than a vector of ones - an arm that is bit-identical to production is a better
    control than one that merely should be.

    A floor is applied because 5.5% of targets are exactly zero: at alpha >= 1 those rows
    would get zero weight and drop out of training entirely. The metric does ignore them,
    but they still carry information about where the boundary between moving and not
    moving lies, and silently discarding 70k rows is a different experiment from
    reweighting.
    """
    if alpha == 0.0:
        return None
    w = np.abs(y) ** alpha
    floor = np.quantile(w[w > 0], 0.01)
    w = np.maximum(w, floor)
    return w / w.mean()


def cv(df: pd.DataFrame, alpha: float, *, seed: int, folds: int = FOLDS) -> list[float]:
    import lightgbm as lgb

    cols = feature_columns(df)
    months, y = df["month"].to_numpy(), df["target"].to_numpy()
    params = {**DEFAULT_PARAMS, "seed": seed, "bagging_seed": seed,
              "feature_fraction_seed": seed}

    def cosine_eval(pred, dataset):
        return "cosine", cosine_similarity(dataset.get_label(), pred), True

    scores = []
    for fold, tr, va in list(iter_folds(months))[-folds:]:
        assert_fold_integrity(months, fold, tr, va)
        w = weights(y[tr], alpha)
        dtr = lgb.Dataset(df.iloc[tr][cols], label=y[tr], weight=w)
        # The VALIDATION set stays unweighted: early stopping must track the metric as it
        # will actually be scored, not the reweighted proxy being trained on.
        dva = lgb.Dataset(df.iloc[va][cols], label=y[va], reference=dtr)
        booster = lgb.train(
            params, dtr, num_boost_round=ROUNDS, valid_sets=[dva], feval=cosine_eval,
            callbacks=[lgb.early_stopping(EARLY_STOPPING, first_metric_only=True,
                                          verbose=False)],
        )
        scores.append(cosine_similarity(y[va], booster.predict(df.iloc[va][cols])))
    return scores


def run(*, seeds: tuple[int, ...] = (0, 1), folds: int = FOLDS,
        alphas: tuple[float, ...] = ALPHAS) -> dict:
    cfg = load_config()
    df = load_dataset("train")
    log.info("%s rows, %d features, alphas %s, %d folds, seeds %s",
             f"{len(df):,}", len(feature_columns(df)), list(alphas), folds, list(seeds))
    zero = float((df["target"] == 0).mean())
    log.info("exactly-zero targets: %.1f%% - these contribute nothing to cosine", zero * 100)

    rows = []
    for seed in seeds:
        by_alpha = {}
        for a in alphas:
            t0 = time.perf_counter()
            by_alpha[a] = cv(df, a, seed=seed, folds=folds)
            log.info("  seed %d  alpha %.1f  cosine %+.5f   [%.0f min]", seed, a,
                     np.mean(by_alpha[a]), (time.perf_counter() - t0) / 60)
        for a in alphas:
            for i, s in enumerate(by_alpha[a]):
                rows.append({"seed": seed, "alpha": a, "fold": i, "cosine": s,
                             "diff_vs_base": s - by_alpha[0.0][i]})

    out = pd.DataFrame(rows)
    fold_noise = 0.0041
    summary = []
    for a in alphas:
        d = out.loc[out.alpha == a, "diff_vs_base"].to_numpy()
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if a != 0.0 else 0.0
        summary.append({"alpha": a, "cosine": float(out.loc[out.alpha == a, "cosine"].mean()),
                        "gain": float(d.mean()), "se": se,
                        "ci_low": float(d.mean() - 1.96 * se),
                        "ci_high": float(d.mean() + 1.96 * se),
                        "improved": int((d > 0).sum()), "n": len(d)})
    summ = pd.DataFrame(summary)

    log.info("\n%s", summ.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    best = summ.loc[summ.gain.idxmax()]
    log.info("bar: %+.5f (fold-to-fold noise)", fold_noise)
    verdict = ("metric alignment PAYS" if best.ci_low > fold_noise
               else "no gain beyond the noise floor")
    log.info("VERDICT: %s", verdict)

    res = {"alphas": list(alphas), "seeds": list(seeds), "folds": folds,
           "fold_noise": fold_noise, "zero_target_share": zero,
           "summary": summ.to_dict("records"), "verdict": verdict}
    dst = Path(cfg.paths.features)
    out.to_csv(dst / "metric_alignment.csv", index=False)
    (dst / "metric_alignment_meta.json").write_text(json.dumps(res, indent=2),
                                                    encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Weight the loss the way cosine weights rows")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--folds", type=int, default=FOLDS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(seeds=tuple(args.seeds), folds=args.folds)


if __name__ == "__main__":
    main()
