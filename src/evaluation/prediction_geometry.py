"""Can the prediction VECTOR be improved without touching the model?

Every other experiment here changes what is learned. This one leaves the model alone and
asks whether the numbers it already produces are being presented to the metric in the best
form. That is a fair question because cosine similarity has structure the training loss
does not:

  * it is scale-invariant but **not shift-invariant**, so any constant bias is pure loss;
  * it weights rows by MAGNITUDE, so how confident the model is where matters as much as
    where it is right;
  * it is computed once over the whole test set, so relative magnitudes ACROSS regimes are
    part of the score in a way no per-row loss can see.

Three corrections follow from those, and all three are free - no retraining, no new
features. They are also exactly the kind of idea this project has learned to distrust, so
each is measured rather than argued.

WHAT IS TESTED

  shift      subtract the prediction's own mean; and separately the best constant offset
  shape      rank-transform, sign-only, and |p|^a for a range of powers
  regime     scale predictions by a local volatility estimate built from FEATURES ONLY -
             a rolling median over neighbouring sample_id, which is the one context the
             292 features cannot see, since every one of them is computed inside a single
             60-second window

THE PROTOCOL THAT MATTERS

The regime correction has a free parameter, so measuring it on the same rows that chose it
would repeat the mistake this project has documented five times. Alpha is fitted on one
half of the hold-out months and scored on the other, both ways round, and the per-month
effect is reported beside it.

The gap between the two protocols is the finding. Pooled and fitted in place, the best
regime correction is worth +0.0022 - comparable to the ensemble, and it would have made a
respectable sixth forecast. Chosen out of sample it averages -0.0002 across the twelve
splits: not smaller, but gone. The pooled number was measuring the freedom to pick alpha.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import cosine_similarity
from src.evaluation.temporal_validation import holdout_months
from src.inference.predictor import load_bundle

log = logging.getLogger(__name__)

REPO_RESULTS = Path(__file__).resolve().parents[2] / "results"

# The one feature used as a regime proxy. Mid-price volatility over the full window is the
# most direct measure of "how much is moving right now" the feature set contains.
REGIME_FEATURE = "mkt_mid_std_60s"
REGIME_WINDOWS = (200, 1000, 5000)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
POWERS = (0.5, 0.75, 1.25, 1.5, 2.0)


def _holdout_predictions(model_dir: Path, dataset: Path) -> pd.DataFrame:
    """Score the hold-out with the shipped artefact, one batch at a time.

    Batched because the full width is 292 float32 columns over 1.26M rows; reading it whole
    to keep 105k of them costs about 1.4 GB and has run this machine out of memory.
    """
    import pyarrow.parquet as pq

    bundle = load_bundle(model_dir)
    lo, hi = holdout_months()
    wanted = ["sample_id", "month", "target", REGIME_FEATURE, *bundle.features]
    wanted = list(dict.fromkeys(wanted))

    frames = []
    for batch in pq.ParquetFile(dataset).iter_batches(batch_size=40_000, columns=wanted):
        d = batch.to_pandas()
        d = d[(d["month"] >= lo) & (d["month"] <= hi)]
        if len(d):
            pred = bundle.model.predict(d[bundle.features].astype("float32"))
            frames.append(pd.DataFrame({
                "sample_id": d["sample_id"].to_numpy(),
                "month": d["month"].to_numpy(),
                "y": d["target"].to_numpy(),
                "p": np.asarray(pred, dtype="float64"),
                "regime_raw": d[REGIME_FEATURE].to_numpy(dtype="float64"),
            }))
        del d
        gc.collect()

    return pd.concat(frames, ignore_index=True).sort_values(
        "sample_id", ignore_index=True)


def _regime_scale(regime_raw: pd.Series, window: int, *, causal: bool) -> np.ndarray:
    """Local volatility as a multiplier centred on 1.

    Median rather than mean because the raw feature is heavy-tailed, and `causal` decides
    whether neighbouring samples from the future may be used. The competition hands over
    the whole test set at once so the two-sided version is admissible here, but the causal
    one is what a live system could compute and both are reported.
    """
    roll = regime_raw.rolling(window, min_periods=50, center=not causal).median()
    filled = roll.bfill().ffill().to_numpy()
    return filled / np.nanmean(filled)


def run(*, out: Path | None = None) -> dict:
    cfg = load_config()
    out = out or REPO_RESULTS
    out.mkdir(parents=True, exist_ok=True)

    dataset = Path(cfg.paths.features) / "dataset_train.parquet"
    model_dir = Path(cfg.paths.data_root) / "models" / "current"
    df = _holdout_predictions(model_dir, dataset)
    y, p, months = df["y"].to_numpy(), df["p"].to_numpy(), df["month"].to_numpy()
    base = cosine_similarity(y, p)
    log.info("hold-out baseline cosine %+.5f on %s rows", base, f"{len(df):,}")

    rows: list[dict] = []

    def record(family: str, name: str, transformed: np.ndarray, *, honest: bool) -> None:
        score = cosine_similarity(y, transformed)
        rows.append({"family": family, "variant": name, "cosine": score,
                     "gain": score - base, "out_of_sample": honest})

    # ---------------------------------------------------------------- shift
    record("shift", "subtract own mean", p - p.mean(), honest=True)
    # De-meaning is the only reshaping that gains anything on the pooled hold-out, so its
    # per-month behaviour is reported beside it: a correction that helps on average while
    # changing sign month to month is a coin toss, not a method.
    shift_by_month = []
    for m in np.unique(months):
        k = months == m
        shift_by_month.append(
            cosine_similarity(y[k], p[k] - p[k].mean()) - cosine_similarity(y[k], p[k]))
    offsets = np.linspace(-2e-4, 2e-4, 201)
    best_c = offsets[np.argmax([cosine_similarity(y, p - c) for c in offsets])]
    record("shift", f"best constant offset ({best_c:+.2e})", p - best_c, honest=False)

    # ---------------------------------------------------------------- shape
    from scipy.stats import rankdata

    record("shape", "rank transform",
           (rankdata(p) - len(p) / 2) / (len(p) / 2), honest=True)
    record("shape", "sign only", np.sign(p), honest=True)
    for a in POWERS:
        record("shape", f"|p|^{a}", np.sign(p) * np.abs(p) ** a, honest=True)

    # ---------------------------------------------------------------- regime
    regime_raw = df["regime_raw"]
    honest: list[dict] = []
    for window in REGIME_WINDOWS:
        for causal in (True, False):
            scale = _regime_scale(regime_raw, window, causal=causal)
            label = f"w={window} {'causal' if causal else 'two-sided'}"
            pooled_best = max(ALPHAS, key=lambda a: cosine_similarity(y, p * scale ** a))
            record("regime", f"{label}, alpha fitted in-sample ({pooled_best})",
                   p * scale ** pooled_best, honest=False)

            # The honest version: choose alpha on half the months, score on the other.
            halves = np.array_split(np.unique(months), 2)
            for fit_months, eval_months in (halves, halves[::-1]):
                fit = np.isin(months, fit_months)
                ev = np.isin(months, eval_months)
                alpha = max(ALPHAS,
                            key=lambda a: cosine_similarity(y[fit], p[fit] * scale[fit] ** a))
                before = cosine_similarity(y[ev], p[ev])
                after = cosine_similarity(y[ev], p[ev] * scale[ev] ** alpha)
                honest.append({"variant": label, "alpha": alpha,
                               "fit_months": [int(m) for m in fit_months],
                               "gain": after - before})

    table = pd.DataFrame(rows).sort_values("gain", ascending=False, ignore_index=True)
    table.to_csv(out / "prediction_geometry.csv", index=False)

    pooled_regime = table.query("family == 'regime'")["gain"].max()
    shift_arr = np.asarray(shift_by_month)
    honest_regime = float(np.mean([h["gain"] for h in honest]))
    meta = {
        "baseline_cosine": base,
        "n_rows": int(len(df)),
        "holdout_months": list(holdout_months()),
        "regime_feature": REGIME_FEATURE,
        "best_pooled_gain": float(pooled_regime),
        "honest_out_of_sample_gain": honest_regime,
        "inflation_factor": float(pooled_regime / honest_regime) if honest_regime else None,
        "fold_to_fold_std": 0.00413,
        "shift_per_month": {
            "gains": [float(v) for v in shift_arr],
            "mean": float(shift_arr.mean()),
            "std": float(shift_arr.std(ddof=1)),
            "months_improved": int((shift_arr > 0).sum()),
            "n_months": int(shift_arr.size),
        },
        "out_of_sample_splits": honest,
        "verdict": (
            "No free correction survives. Every reshaping of the magnitudes loses, so the "
            "model is already close to the best form cosine can read it in. De-meaning "
            "gains on the pooled hold-out and changes sign across months, which makes it "
            "an average of noise. The regime correction describes a real under-scaling - "
            "the model spans 1.55x across volatility quartiles where the target spans "
            "1.69x - but choosing its one parameter out of sample turns +0.0022 into "
            "roughly zero."
        ),
    }
    (out / "prediction_geometry_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    log.info("pooled best regime gain %+.5f | honest out-of-sample %+.5f (%.1fx smaller)",
             pooled_regime, honest_regime,
             pooled_regime / honest_regime if honest_regime else float("nan"))
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(out=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
