"""How cosine weights subgroups - and why the earlier reweighting prediction was wrong.

THE IDENTITY

Cosine similarity is computed over the pooled vector, without centring:

    cos(y, p) = <y, p> / (||y|| * ||p||)

Split the samples into disjoint groups g. Because the inner product is additive over the
partition, this factors EXACTLY:

    cos(y, p) = SUM_g  cos_g * w_g        with   w_g = ||y_g|| * ||p_g|| / (||y|| * ||p||)

so a pooled cosine is a weighted average of per-group cosines whose weights are products
of MAGNITUDES, not sample counts. The weights sum to at most 1 (Cauchy-Schwarz), with
equality only when every group has the same cosine.

WHY THIS MATTERS HERE

Notebook 04 forecast a leaderboard score by reweighting per-quartile hold-out cosines by
the test set's *sample shares*. That is the wrong weighting for this metric: it silently
assumes ||y_g|| * ||p_g|| is proportional to n_g, i.e. that target and prediction
magnitudes are constant across groups. In a financial target they are not - the spread
quartiles differ in volatility, which is exactly what cosine is sensitive to.

So the earlier prediction was not merely incomplete; it was mis-specified. This module
measures the size of that error.

WHAT CANNOT BE FIXED

The corrected weights need ||y_g|| on the TEST set, and the test targets are unobservable.
So the honest conclusion is not a better point prediction, but a statement about which
predictions are computable at all: with cosine, a subgroup-reweighting forecast requires
an assumption about unobserved target magnitudes. Count-weighting is one such assumption,
and a bad one. This module reports the alternative and its sensitivity instead of
pretending the number is knowable.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import cosine_similarity

log = logging.getLogger(__name__)

SPREAD_COL = "mkt_rel_spread_last"
HOLDOUT = (65, 70)


def decompose(y: np.ndarray, pred: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    """Exact per-group decomposition of a pooled cosine.

    Returns one row per group with its own cosine, the magnitude weight the metric
    actually applies, and the count weight a naive analysis would assume.
    """
    y, pred, group = np.asarray(y), np.asarray(pred), np.asarray(group)
    ny, npd = np.linalg.norm(y), np.linalg.norm(pred)
    rows = []
    for g in pd.unique(group):
        m = group == g
        yg, pg = y[m], pred[m]
        rows.append({
            "group": g,
            "n": int(m.sum()),
            "cosine": cosine_similarity(yg, pg),
            "y_norm": float(np.linalg.norm(yg)),
            "pred_norm": float(np.linalg.norm(pg)),
            "y_rms": float(np.sqrt(np.mean(yg ** 2))),
            "weight": float(np.linalg.norm(yg) * np.linalg.norm(pg) / (ny * npd)),
        })
    out = pd.DataFrame(rows)
    out["count_weight"] = out["n"] / out["n"].sum()
    out["contribution"] = out["cosine"] * out["weight"]
    return out.sort_values("group").reset_index(drop=True)


def verify_identity(y: np.ndarray, pred: np.ndarray, group: np.ndarray) -> dict:
    """The decomposition must reproduce the pooled cosine to floating-point precision."""
    parts = decompose(y, pred, group)
    pooled = cosine_similarity(y, pred)
    rebuilt = float(parts.contribution.sum())
    return {
        "pooled": pooled, "rebuilt": rebuilt, "abs_error": abs(pooled - rebuilt),
        "weight_sum": float(parts.weight.sum()),
    }


def holdout_predictions(model_dir: str | Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Re-score the hold-out months with the saved artefact."""
    from src.inference.predictor import load_bundle
    from src.models.train import load_dataset

    cfg = load_config()
    bundle = load_bundle(model_dir or Path(cfg.paths.data_root) / "models" / "current")
    df = load_dataset("train")
    df = df[df.month.between(*HOLDOUT)].reset_index(drop=True)
    pred = bundle.model.predict(df[bundle.features])
    return df, np.asarray(pred, dtype=float)


def spread_buckets(values: pd.Series, edges: np.ndarray | None = None):
    labels = ["Q1 (tightest)", "Q2", "Q3", "Q4 (widest)"]
    if edges is None:
        cats, edges = pd.qcut(values, 4, labels=labels, retbins=True)
        return cats, edges
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True), edges


def corrected_forecast(parts: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """Redo the leaderboard forecast with the weighting the metric actually uses.

    The magnitude weight is w_g ~ ||y_g|| * ||p_g||. On the test set ||p_g|| is directly
    observable - the submitted predictions are in hand - and only ||y_g|| is not. Rather
    than assume it away, it is carried over from the hold-out as a per-bucket target RMS,
    which is a far weaker assumption than count-weighting: the hold-out RMS varies only
    about 7% across buckets, while the weights themselves vary by 50%.

    Returns the test-side bucket table; the caller compares the resulting forecasts.
    """
    import pyarrow.parquet as pq

    cfg = load_config()
    feat = Path(cfg.paths.features)
    tf = pq.read_table(feat / "dataset_test.parquet",
                       columns=["sample_id", SPREAD_COL]).to_pandas()
    sub = pd.read_csv(feat / "submission.csv")
    test = tf.merge(sub, on="sample_id", how="inner")
    test["bucket"] = spread_buckets(test[SPREAD_COL], edges)[0].astype(str)

    ref = parts.set_index("group")
    rows = []
    for g, sub_g in test.groupby("bucket"):
        if g not in ref.index:
            continue
        n = len(sub_g)
        rows.append({
            "group": g, "n": n,
            "pred_norm": float(np.linalg.norm(sub_g.prediction.to_numpy())),
            # ||y_g|| estimated as (hold-out RMS for this bucket) * sqrt(n)
            "y_norm_est": float(ref.loc[g, "y_rms"] * np.sqrt(n)),
            "holdout_cosine": float(ref.loc[g, "cosine"]),
        })
    out = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    out["count_weight"] = out.n / out.n.sum()
    raw = out.y_norm_est * out.pred_norm
    out["weight"] = raw / raw.sum()
    return out


def run(model_dir: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    df, pred = holdout_predictions(model_dir)
    y = df["target"].to_numpy(dtype=float)
    buckets, edges = spread_buckets(df[SPREAD_COL])
    parts = decompose(y, pred, buckets.astype(str).to_numpy())

    ident = verify_identity(y, pred, buckets.astype(str).to_numpy())
    log.info("identity check: pooled %.10f  rebuilt %.10f  error %.2e  weights sum to %.6f",
             ident["pooled"], ident["rebuilt"], ident["abs_error"], ident["weight_sum"])
    log.info("\n%s", parts[["group", "n", "cosine", "y_rms", "count_weight", "weight"]]
             .to_string(index=False, float_format=lambda v: f"{v:,.5f}"))

    # What the two weightings say about the SAME hold-out numbers.
    by_count = float((parts.count_weight * parts.cosine).sum())
    by_magnitude = float((parts.weight * parts.cosine).sum())
    log.info("hold-out, count-weighted     %+.5f   <- what notebook 04 assumed", by_count)
    log.info("hold-out, magnitude-weighted %+.5f   <- what the metric actually does",
             by_magnitude)
    log.info("pooled cosine (ground truth) %+.5f", ident["pooled"])

    # The forecast, redone with each weighting.
    fc = corrected_forecast(parts[parts.group != "nan"], edges)
    fc_count = float((fc.count_weight * fc.holdout_cosine).sum())
    fc_magnitude = float((fc.weight * fc.holdout_cosine).sum())
    log.info("\n%s", fc[["group", "n", "holdout_cosine", "count_weight", "weight"]]
             .to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    log.info("forecast, count-weighted     %+.5f   <- notebook 04's method", fc_count)
    log.info("forecast, magnitude-weighted %+.5f   <- corrected", fc_magnitude)
    log.info("actual leaderboard           %+.5f", 0.128)

    dst = Path(cfg.paths.features)
    fc.to_csv(dst / "cosine_forecast_corrected.csv", index=False)
    parts.to_csv(dst / "cosine_decomposition.csv", index=False)
    (dst / "cosine_decomposition_meta.json").write_text(json.dumps({
        "holdout_months": list(HOLDOUT), "spread_col": SPREAD_COL,
        "bucket_edges": [float(e) for e in edges],
        "count_weighted": by_count, "magnitude_weighted": by_magnitude,
        "forecast_count_weighted": fc_count, "forecast_magnitude_weighted": fc_magnitude,
        "actual_leaderboard": 0.128, **ident,
    }, indent=2), encoding="utf-8")
    return parts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Decompose the pooled cosine by subgroup")
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.model_dir)


if __name__ == "__main__":
    main()
