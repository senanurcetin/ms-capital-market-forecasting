"""Feature-family ablation: what does each source table actually buy?

SHAP answers "which features does the model lean on". That is not the same question as
"what would I lose without them" - SHAP importance is computed on a model that already
has every feature, so a family can rank high simply by being a convenient encoding of
information also present elsewhere.

Ablation answers the second question directly: retrain on subsets and compare.

Two quantities are reported, and they differ in a way that matters:

  standalone     what a family achieves on its own
  marginal       what it adds on top of everything else  (all - all_without_family)

A family can be strong standalone yet worth little at the margin, if the others already
carry its information. That is the case worth catching, because it is the case where a
whole ingestion path could be dropped.

Runs on a sample by default: the comparison is relative, every subset gets identical
treatment, and the full-data version would take hours for no change in ranking.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import cosine_similarity
from src.evaluation.temporal_validation import iter_folds, stability
from src.models.base import NON_FEATURES
from src.models.train import assert_fold_integrity, load_dataset

log = logging.getLogger(__name__)

FAMILIES = ("mkt", "ord", "txn")


def family_of(column: str) -> str:
    return column.split("_", 1)[0]


def subsets(families: tuple[str, ...] = FAMILIES) -> list[tuple[str, ...]]:
    """Every non-empty combination, smallest first."""
    out: list[tuple[str, ...]] = []
    for r in range(1, len(families) + 1):
        out.extend(itertools.combinations(families, r))
    return out


def run(
    df: pd.DataFrame | None = None,
    *,
    sample_frac: float = 0.25,
    rounds: int = 400,
    early_stopping: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    import lightgbm as lgb

    cfg = load_config()
    if df is None:
        df = load_dataset("train")
    if sample_frac and sample_frac < 1.0:
        df = df.groupby("month", group_keys=False).sample(frac=sample_frac, random_state=seed)
    log.info("ablation on %s rows (sample_frac=%s)", f"{len(df):,}", sample_frac)

    all_features = [c for c in df.columns if c not in NON_FEATURES]
    by_family = {f: [c for c in all_features if family_of(c) == f] for f in FAMILIES}
    for f, cols in by_family.items():
        log.info("  %s: %d features", f, len(cols))

    months = df["month"].to_numpy()
    y = df["target"].to_numpy()
    params = {
        "objective": "regression", "metric": "None", "learning_rate": 0.05,
        "num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
        "max_bin": 127, "verbosity": -1, "seed": seed, "num_threads": 0,
    }

    def cosine_eval(pred, dataset):
        return "cosine", cosine_similarity(dataset.get_label(), pred), True

    rows = []
    for combo in subsets():
        cols = [c for f in combo for c in by_family[f]]
        scores = []
        t0 = time.perf_counter()
        for fold, tr, va in iter_folds(months):
            assert_fold_integrity(months, fold, tr, va)
            dtr = lgb.Dataset(df.iloc[tr][cols], label=y[tr])
            dva = lgb.Dataset(df.iloc[va][cols], label=y[va], reference=dtr)
            booster = lgb.train(
                params, dtr, num_boost_round=rounds, valid_sets=[dva], feval=cosine_eval,
                callbacks=[lgb.early_stopping(early_stopping, first_metric_only=True,
                                              verbose=False)],
            )
            scores.append(cosine_similarity(y[va], booster.predict(df.iloc[va][cols])))
        st = stability(scores)
        rows.append({
            "families": "+".join(combo), "n_families": len(combo), "n_features": len(cols),
            "cosine_mean": st["mean"], "cosine_std": st["std"], "cosine_min": st["min"],
            "seconds": round(time.perf_counter() - t0, 1),
        })
        log.info("  %-15s %3d features  cosine %+.5f (std %.5f)  %.0fs",
                 rows[-1]["families"], len(cols), st["mean"], st["std"], rows[-1]["seconds"])

    out = pd.DataFrame(rows).sort_values("cosine_mean", ascending=False).reset_index(drop=True)

    # Marginal value: full set minus the full set without that family.
    full = out.loc[out.families == "+".join(FAMILIES), "cosine_mean"].iloc[0]
    marginal = []
    for f in FAMILIES:
        without = "+".join(x for x in FAMILIES if x != f)
        w = out.loc[out.families == without, "cosine_mean"].iloc[0]
        standalone = out.loc[out.families == f, "cosine_mean"].iloc[0]
        marginal.append({
            "family": f, "n_features": len(by_family[f]),
            "standalone": standalone, "without_it": w, "marginal_gain": full - w,
            "share_of_full": standalone / full,
        })
    marg = pd.DataFrame(marginal).sort_values("marginal_gain", ascending=False)

    dst = Path(cfg.paths.features)
    out.to_csv(dst / "ablation_subsets.csv", index=False)
    marg.to_csv(dst / "ablation_marginal.csv", index=False)
    log.info("\n%s", out.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    log.info("\nmarginal value of each family (full set = %+.5f):\n%s", full,
             marg.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    (dst / "ablation_meta.json").write_text(json.dumps({
        "sample_frac": sample_frac, "rounds": rounds, "n_rows": len(df),
        "full_set_cosine": full,
    }, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Feature-family ablation")
    ap.add_argument("--sample-frac", type=float, default=0.25)
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    np.random.seed(0)
    run(sample_frac=args.sample_frac, rounds=args.rounds)


if __name__ == "__main__":
    main()
