"""How many of the 292 features are actually distinct?

This exists because an audit of a NEW feature set found five of eighteen columns
duplicating something that already existed - one of them at correlation 1.000. The obvious
next question was whether the original 292 had the same problem, and they do.

WHAT IT FINDS

31 pairs correlate above 0.999, and several are exactly 1.000. They fall into three
kinds, and each is a small lesson rather than a bug:

  rate vs count over a FIXED window
      txn_intensity_60s == txn_n_total, ord_event_rate_60s == ord_n_total,
      mkt_snapshot_rate_600s == mkt_n_snapshots. Dividing a count by a constant is not a
      second feature.

  a window that IS the full window
      txn_vwap_60s == txn_vwap_total, mkt_spread_mean_600s == mkt_spread_mean_clean.
      The widest nested window coincides with the whole sample by construction.

  a normalisation that normalises by 1
      mkt_spread_mean_5s == mkt_rel_spread_mean_5s. Prices are normalised so that mid is
      about 1.0 - a fact this project measured and documented early - so dividing spread
      by mid changes nothing. The measurement was made and its consequence for the feature
      set was never followed up.

WHAT IT DOES NOT MEAN

No result is invalidated. Gradient boosting is untroubled by correlated inputs, and the
ablation, SHAP and drift analyses are all unaffected. What is affected is a headline:
"292 features" counts columns, not information, and the honest count is lower.

Whether pruning would raise the score is a separate question, and given a fold-to-fold
noise of 0.0041 the answer is probably no - but that would need measuring rather than
assuming, which is the whole habit here.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.base import feature_columns

log = logging.getLogger(__name__)

NEAR_DUPLICATE = 0.999


def correlation_pairs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Every pair of features with their absolute correlation, highest first.

    NaN and infinities are zeroed before standardising: the feature layer emits NaN by
    design for empty short windows, and a single infinity would make a whole column's
    correlation undefined.
    """
    X = np.nan_to_num(df[cols].to_numpy(dtype=np.float32), nan=0.0,
                      posinf=0.0, neginf=0.0)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    del X
    C = np.abs(Z.T @ Z) / len(Z)
    np.fill_diagonal(C, 0.0)
    iu = np.triu_indices(len(cols), 1)
    r = C[iu]
    order = np.argsort(r)[::-1]
    return pd.DataFrame({
        "a": [cols[iu[0][k]] for k in order],
        "b": [cols[iu[1][k]] for k in order],
        "abs_corr": r[order],
    })


def effective_count(pairs: pd.DataFrame, cols: list[str],
                    threshold: float = NEAR_DUPLICATE) -> int:
    """Distinct features left after collapsing each near-duplicate group to one.

    Union-find rather than a pairwise count: if a == b and b == c, that is one feature
    surviving out of three, not two removals counted twice.
    """
    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs.loc[pairs.abs_corr > threshold, ["a", "b"]].itertuples(index=False):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(c) for c in cols})


def run(*, n_sample: int = 150_000, seed: int = 0) -> dict:
    from src.models.train import load_dataset

    cfg = load_config()
    df = load_dataset("train").sample(n=n_sample, random_state=seed)
    cols = feature_columns(df)
    pairs = correlation_pairs(df, cols)
    del df

    n_eff = effective_count(pairs, cols)
    counts = {f"pairs_above_{t}": int((pairs.abs_corr > t).sum())
              for t in (0.999, 0.99, 0.95)}

    log.info("features as counted      %d", len(cols))
    log.info("effective (|r| <= %.3f)  %d", NEAR_DUPLICATE, n_eff)
    for k, v in counts.items():
        log.info("  %-20s %d", k, v)
    log.info("\n%s", pairs.head(15).to_string(index=False,
                                              float_format=lambda v: f"{v:,.4f}"))

    res = {"n_features": len(cols), "n_effective": n_eff,
           "threshold": NEAR_DUPLICATE, "n_sample": n_sample, **counts}
    dst = Path(cfg.paths.features)
    pairs.head(200).to_csv(dst / "feature_redundancy.csv", index=False)
    (dst / "feature_audit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="How many features are actually distinct?")
    ap.add_argument("--sample", type=int, default=150_000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(n_sample=args.sample)


if __name__ == "__main__":
    main()
